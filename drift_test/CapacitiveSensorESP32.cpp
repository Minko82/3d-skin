/*
CapacitiveSensorESP32.cpp and CapacitiveSensorESP32.h

Adapted from CapacitiveSensorR4 by Takanori Fujiwara and S. Sandra Bae
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

Original CapacitiveSensorR4 Copyright (c) 2023, Takanori Fujiwara and S. Sandra Bae
Original CapacitiveSensor Library Copyright (c) 2009 Paul Bagder
https://github.com/PaulStoffregen/CapacitiveSensor

Adapted for ESP32-C6 (and compatible ESP32 variants).
Uses standard Arduino digitalWrite/digitalRead/pinMode — compatible with
the ESP32 Arduino core. noInterrupts()/interrupts() disable the FreeRTOS
scheduler briefly during each charge cycle; each critical section is
microseconds long so this is safe.
*/

#include "Arduino.h"
#include "CapacitiveSensorESP32.h"

CapacitiveSensor::CapacitiveSensor(uint8_t sendPin, uint8_t receivePin)
{
    sPin = sendPin;
    rPin = receivePin;

    set_CS_Timeout_Millis(2000);
    set_CS_AutocaL_Millis(0xFFFFFFFF);

    if (sendPin >= NUM_DIGITAL_PINS)
        error = -1;
    if (receivePin >= NUM_DIGITAL_PINS)
        error = -1;

    pinMode(sendPin, OUTPUT);
    pinMode(receivePin, INPUT);
    digitalWrite(sendPin, LOW);

    leastTotal = 0x0FFFFFFFL;
    lastCal = millis();
}

long CapacitiveSensor::capacitiveSensor(uint8_t samples)
{
    total = 0;
    if (samples == 0)
        return 0;
    if (error < 0)
        return -1;

    for (uint8_t i = 0; i < samples; i++)
    {
        if (SenseOneCycle() < 0)
            return -2;
    }

    unsigned long diff = (total > leastTotal) ? total - leastTotal : leastTotal - total;
    if ((millis() - lastCal > CS_AutocaL_Millis) && diff < (int)(.10 * (float)leastTotal))
    {
        leastTotal = 0x0FFFFFFFL;
        lastCal = millis();
    }

    if (total < leastTotal)
        leastTotal = total;

    return (total - leastTotal);
}

long CapacitiveSensor::capacitiveSensorRaw(uint8_t samples)
{
    total = 0;
    if (samples == 0)
        return 0;
    if (error < 0)
        return -1;

    for (uint8_t i = 0; i < samples; i++)
    {
        if (SenseOneCycle() < 0)
            return -2;
    }

    return total;
}

int CapacitiveSensor::SenseOneCycle(void)
{
    noInterrupts();
    digitalWrite(sPin, LOW);
    pinMode(rPin, INPUT);
    pinMode(rPin, OUTPUT);
    digitalWrite(rPin, LOW);
    delayMicroseconds(100);
    pinMode(rPin, INPUT);
    digitalWrite(sPin, HIGH);
    interrupts();

    while (!digitalRead(rPin) && (total < CS_Timeout_Millis))
        total++;

    if (total > CS_Timeout_Millis)
        return -2;

    noInterrupts();
    digitalWrite(rPin, HIGH);
    pinMode(rPin, OUTPUT);
    digitalWrite(rPin, HIGH);
    pinMode(rPin, INPUT);
    digitalWrite(sPin, LOW);
    interrupts();

    while (digitalRead(rPin) && (total < CS_Timeout_Millis))
        total++;

    if (total >= CS_Timeout_Millis)
        return -2;
    else
        return 1;
}
