/*
CapacitiveSensorESP32.cpp and CapacitiveSensorESP32.h

Adapted from CapacitiveSensorR4 by Takanori Fujiwara and S. Sandra Bae
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

Original CapacitiveSensorR4 Copyright (c) 2023, Takanori Fujiwara and S. Sandra Bae
Original CapacitiveSensor Library Copyright (c) 2009 Paul Bagder
https://github.com/PaulStoffregen/CapacitiveSensor

Adapted for ESP32-C6 (and compatible ESP32 variants).
loopTimingFactor is set for ESP32-C6 @ 160 MHz — recalibrate empirically
if readings drift or saturate on your board.
*/

#ifndef CapacitiveSensorESP32_h
#define CapacitiveSensorESP32_h

class CapacitiveSensor
{
public:
    CapacitiveSensor(uint8_t sendPin, uint8_t receivePin);
    long capacitiveSensorRaw(uint8_t samples);
    long capacitiveSensor(uint8_t samples);
    void set_CS_Timeout_Millis(unsigned long timeout_millis)
    {
        CS_Timeout_Millis = (timeout_millis * (float)loopTimingFactor * (float)F_CPU) / 16000000;
    };
    void reset_CS_AutoCal() { leastTotal = 0x0FFFFFFFL; };
    void set_CS_AutocaL_Millis(unsigned long autoCal_millis) { CS_AutocaL_Millis = autoCal_millis; };

private:
    const unsigned int loopTimingFactor = 1000; // empirically set for ESP32-C6 @ 160 MHz
    int error = 1;
    unsigned long leastTotal;
    unsigned long CS_Timeout_Millis;
    unsigned long CS_AutocaL_Millis;
    unsigned long lastCal;
    unsigned long total;
    uint8_t sPin;
    uint8_t rPin;
    int SenseOneCycle(void);
};

#endif
