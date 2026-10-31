import asyncio
import logging
import voluptuous as vol
from typing import Any, Optional, Tuple

from .idealled import IDEALLEDInstance
from .const import (
    DOMAIN,
    CONF_RESEND_INTERVAL,
    DEFAULT_RESEND_INTERVAL,
    RETRY_MIN_INTERVAL,
    COMMAND_TIMEOUT,
)

from homeassistant.const import CONF_MAC, STATE_ON, STATE_OFF
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.util import dt as dt_util
import homeassistant.helpers.config_validation as cv
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.components.light import (
    PLATFORM_SCHEMA,
    ATTR_BRIGHTNESS,
    ATTR_HS_COLOR,
    ATTR_EFFECT,
    ColorMode,
    LightEntity,
    LightEntityFeature,
)
from homeassistant.util.color import match_max_scale
from homeassistant.helpers import device_registry

LOGGER = logging.getLogger(__name__)
PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend({vol.Required(CONF_MAC): cv.string})

async def async_setup_entry(hass, config_entry, async_add_devices):
    instance = hass.data[DOMAIN][config_entry.entry_id]
    # No initial connect here: with a weak BLE link it can stall HA startup for
    # minutes, and it learns nothing (the device reports no state). The entity
    # re-asserts its restored state in the background once added.
    resend_interval = config_entry.options.get(
        CONF_RESEND_INTERVAL, DEFAULT_RESEND_INTERVAL
    )
    async_add_devices(
        [
            IDEALLEDLight(
                instance,
                config_entry.data["name"],
                config_entry.entry_id,
                resend_interval,
            )
        ]
    )


class IDEALLEDLight(LightEntity, RestoreEntity):
    def __init__(
        self,
        idealledinstance: IDEALLEDInstance,
        name: str,
        entry_id: str,
        resend_interval: int = DEFAULT_RESEND_INTERVAL,
    ) -> None:
        self._instance = idealledinstance
        self._entry_id = entry_id
        self._attr_supported_color_modes = {ColorMode.HS}
        self._attr_supported_features = LightEntityFeature.EFFECT
        #self._attr_brightness_step_pct = 10
        self._attr_name = name
        self._attr_unique_id = self._instance.mac
        self._instance.local_callback = self.light_local_callback
        # The device gives no state feedback, so we remember what HA last asked
        # for and keep re-sending it. None = no command seen yet, don't enforce.
        self._desired_on: bool | None = None
        self._in_sync = False
        self._last_confirmed = None
        self._consecutive_failures = 0
        self._resend_interval = resend_interval
        self._command_task: asyncio.Task | None = None
        self._enforce_task: asyncio.Task | None = None
        self._wake = asyncio.Event()

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        last = await self.async_get_last_state()
        if last is not None and last.state in (STATE_ON, STATE_OFF):
            self._desired_on = last.state == STATE_ON
            if (bri := last.attributes.get(ATTR_BRIGHTNESS)) is not None:
                self._instance._brightness = bri
            if (hs := last.attributes.get(ATTR_HS_COLOR)) is not None:
                self._instance._hs_color = tuple(hs)
            LOGGER.debug(
                "%s: restored desired state %s", self.name, last.state
            )
        if self._resend_interval:
            self._enforce_task = self.hass.async_create_background_task(
                self._enforce_loop(), f"ideal_led enforce {self._instance.mac}"
            )
            # Assert the restored state straight away rather than waiting.
            self._wake.set()

    async def async_will_remove_from_hass(self) -> None:
        for task in (self._enforce_task, self._command_task):
            if task and not task.done():
                task.cancel()
        await super().async_will_remove_from_hass()

    def _run_command(self, make_coro) -> asyncio.Task:
        """Run a device command in the background, superseding any in-flight one.

        Service calls return immediately so a slow/unreachable strip can't block
        the calling automation (which would drop later triggers in mode: single).
        """
        if self._command_task and not self._command_task.done():
            self._command_task.cancel()
        self._command_task = self.hass.async_create_task(self._guarded(make_coro))
        return self._command_task

    async def _guarded(self, make_coro) -> None:
        try:
            async with asyncio.timeout(COMMAND_TIMEOUT):
                await make_coro()
        except asyncio.CancelledError:
            raise
        except Exception as err:  # noqa: BLE001 - BLE stack raises all sorts
            self._in_sync = False
            self._consecutive_failures += 1
            LOGGER.warning(
                "%s: failed to send %s (attempt %s), will retry: %s",
                self.name,
                "on" if self._desired_on else "off",
                self._consecutive_failures,
                err or type(err).__name__,
            )
        else:
            self._in_sync = True
            self._consecutive_failures = 0
            self._last_confirmed = dt_util.utcnow()
        # Restart the enforcer's clock: full interval if in sync, backoff if not.
        self._wake.set()
        self.async_write_ha_state()

    def _next_delay(self) -> float:
        if self._in_sync:
            return self._resend_interval
        backoff = RETRY_MIN_INTERVAL * 2 ** max(0, self._consecutive_failures - 1)
        return min(backoff, self._resend_interval)

    async def _enforce_loop(self) -> None:
        while True:
            try:
                async with asyncio.timeout(self._next_delay()):
                    await self._wake.wait()
            except TimeoutError:
                pass
            woken = self._wake.is_set()
            self._wake.clear()
            if self._desired_on is None:
                continue
            if self._command_task is not None:
                # Woken because a command just finished: restart the clock.
                # Or a command is still in flight: leave it be.
                if woken or not self._command_task.done():
                    continue
            # else: timer expired, or first wake after restore (no command yet).
            LOGGER.debug(
                "%s: re-sending desired state %s",
                self.name,
                "on" if self._desired_on else "off",
            )
            task = self._run_command(lambda on=self._desired_on: self._send_power(on))
            await asyncio.wait([task])

    async def _send_power(self, on: bool) -> None:
        if on:
            await self._instance.turn_on()
        else:
            await self._instance.turn_off()

    async def _apply_turn_on(self, kwargs: dict) -> None:
        # Always send the power command: the device may have been switched off
        # behind our back, so our cached is_on can't be trusted to skip it.
        await self._instance.turn_on()

        if ATTR_BRIGHTNESS in kwargs and len(kwargs) == 1:
            # Only brightness changed
            LOGGER.debug(f"Only setting brightness: {kwargs[ATTR_BRIGHTNESS]}")
            await self._instance.set_brightness(kwargs[ATTR_BRIGHTNESS])

        if ATTR_HS_COLOR in kwargs:
            if kwargs[ATTR_HS_COLOR] != self.hs_color:
                bri = kwargs[ATTR_BRIGHTNESS] if ATTR_BRIGHTNESS in kwargs else self._instance._brightness
                LOGGER.debug(f"Setting color: {kwargs[ATTR_HS_COLOR]}. Brightness: {bri}")
                await self._instance.set_rgb_color(kwargs[ATTR_HS_COLOR], bri)

        if ATTR_EFFECT in kwargs:
            if kwargs[ATTR_EFFECT] != self.effect:
                bri = kwargs[ATTR_BRIGHTNESS] if ATTR_BRIGHTNESS in kwargs else self._instance._brightness
                await self._instance.set_effect(kwargs[ATTR_EFFECT], bri)

    @property
    def extra_state_attributes(self):
        return {
            "in_sync": self._in_sync,
            "last_confirmed": self._last_confirmed,
            "consecutive_failures": self._consecutive_failures,
        }

    @property
    def available(self):
        # Always report as available so users can attempt to control the light.
        # The actual connection happens on command, which will succeed once
        # Bluetooth is ready. This avoids the chicken-and-egg problem where
        # an "unavailable" entity can't be controlled to trigger a connection.
        return True

    @property
    def brightness(self):
        return self._instance.brightness

    @property
    def brightness_pct(self):
        return (self._instance.brightness / 255) * 100
    
    @property
    def brightness_step_pct(self):
        return self._attr_brightness_step_pct
    
    @property
    def is_on(self) -> Optional[bool]:
        # Report what we're enforcing; in_sync says whether the device has acked it.
        if self._desired_on is not None:
            return self._desired_on
        return self._instance.is_on

    @property
    def effect_list(self):
        return self._instance.effect_list

    @property
    def effect(self):
        return self._instance._effect

    @property
    def supported_features(self) -> int:
        """Flag supported features."""
        return self._attr_supported_features

    @property
    def supported_color_modes(self) -> int:
        """Flag supported color modes."""
        return self._attr_supported_color_modes

    # @property
    # def rgb_color(self):
    #     """Return the hs color value."""
    #     return self._instance.rgb_color
    
    @property
    def hs_color(self):
        """Return the hs color value."""
        return self._instance.hs_color
    
    @property
    def color_mode(self):
        """Return the color mode of the light."""
        return self._instance._color_mode

    @property
    def firmware_version(self):
        return self._instance.firmware_version
    
    @property
    def device_info(self):
        """Return device info."""
        return DeviceInfo(
            identifiers={
                # Serial numbers are unique identifiers within a specific domain
                (DOMAIN, self._instance.mac)
            },
            name=self.name,
            connections={(device_registry.CONNECTION_NETWORK_MAC, self._instance.mac)},
            model=self._instance._model,
            sw_version=self.firmware_version
        )

    @property
    def should_poll(self):
        return False

    async def async_turn_on(self, **kwargs: Any) -> None:
        LOGGER.info("Turn on called.  kwargs: "+str(kwargs))
        self._desired_on = True
        self._in_sync = False
        self._run_command(lambda: self._apply_turn_on(kwargs))
        self.async_write_ha_state()

    async def async_turn_off(self, **kwargs: Any) -> None:
        self._desired_on = False
        self._in_sync = False
        self._run_command(self._instance.turn_off)
        self.async_write_ha_state()

    async def async_update(self) -> None:
        LOGGER.debug("async update called")
        await self._instance.update()
        self.async_write_ha_state()
    
    def light_local_callback(self):
        self.async_write_ha_state()

    async def update_ha_state(self) -> None:
        await self._instance.update()
        self.async_write_ha_state()
