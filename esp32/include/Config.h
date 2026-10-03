#pragma once
// ===========================================================================
//  Config.h
//  Global configuration constants for the Predictive Maintenance Edge Node.
//
//  All tunable parameters live here so the rest of the code stays clean.
//  Modify this file to adapt the firmware to different hardware or deployments.
// ===========================================================================

#include <cstdint>

// ---------------------------------------------------------------------------
// Node Identity
// ---------------------------------------------------------------------------
/// Unique identifier for this edge node (0–255). Change per deployment.
constexpr uint8_t  NODE_ID          = 1;

/// Binary protocol version transmitted in every packet header.
constexpr uint8_t  PROTOCOL_VERSION = 1;

// ---------------------------------------------------------------------------
// WiFi Credentials
// ---------------------------------------------------------------------------
constexpr char WIFI_SSID[]          = "########";
constexpr char WIFI_PASSWORD[]      = "HHHHHHHH";

/// How long to wait for a WiFi association before retrying (ms).
constexpr uint32_t WIFI_CONNECT_TIMEOUT_MS  = 15000;

/// Delay between WiFi reconnection attempts (ms).
constexpr uint32_t WIFI_RECONNECT_DELAY_MS  = 5000;

// ---------------------------------------------------------------------------
// TCP Server (Raspberry Pi)
// ---------------------------------------------------------------------------
constexpr char     SERVER_IP[]      = "00.000.00.00";
constexpr uint16_t SERVER_PORT      = 9000;

/// How long to wait for a TCP connection to be established (ms).
constexpr uint32_t TCP_CONNECT_TIMEOUT_MS   = 10000;

/// How long to wait for an ACK byte from the server (ms).
constexpr uint32_t TCP_ACK_TIMEOUT_MS       = 3000;

/// Maximum number of retransmission attempts before dropping a packet.
constexpr uint8_t  TCP_MAX_RETRIES          = 5;

/// Delay between retransmission attempts (ms).
constexpr uint32_t TCP_RETRY_DELAY_MS       = 500;

/// ACK byte the server must send to confirm receipt.
constexpr uint8_t  ACK_BYTE                 = 0x06;  // ASCII ACK

// ---------------------------------------------------------------------------
// ADXL345 – SPI Pin Mapping (VSPI bus on ESP32)
// ---------------------------------------------------------------------------
constexpr int PIN_ADXL_CS   = 5;   ///< Chip Select
constexpr int PIN_SPI_MISO  = 19;  ///< MISO (SDO on ADXL345)
constexpr int PIN_SPI_MOSI  = 23;  ///< MOSI (SDA on ADXL345)
constexpr int PIN_SPI_SCK   = 18;  ///< Clock (SCL on ADXL345)

/// SPI clock frequency for ADXL345. Custom driver configured for 1 MHz.
constexpr uint32_t SPI_FREQ_HZ = 1000000;


// ---------------------------------------------------------------------------
// Sampling & Processing Parameters
// ---------------------------------------------------------------------------
/// ADXL345 sampling frequency in Hz (default 500 Hz).
/// Must be ≤ ADXL345 ODR setting (we configure ODR = 800 Hz internally).
constexpr uint32_t SAMPLE_FREQ_HZ      = 500;

/// Length of each acquisition window in seconds.
constexpr uint32_t WINDOW_SECONDS      = 2;

/// Total number of accelerometer samples per window (X, Y, Z each).
constexpr uint32_t SAMPLES_PER_WINDOW  = SAMPLE_FREQ_HZ * WINDOW_SECONDS;  // 1000


// ---------------------------------------------------------------------------
// Buffer Manager
// ---------------------------------------------------------------------------
/// Number of double-buffer slots (must be exactly 2 for ping-pong).
constexpr uint8_t  NUM_BUFFERS = 2;

// ---------------------------------------------------------------------------
// Packet Format Constants
// ---------------------------------------------------------------------------
constexpr uint16_t PACKET_MAGIC = 0xABCD;

// ---------------------------------------------------------------------------
// FreeRTOS Task Priorities & Stack Sizes
// ---------------------------------------------------------------------------
/// Sensor acquisition runs at highest user priority to meet timing.
constexpr UBaseType_t PRIORITY_SENSOR    = 5;
constexpr UBaseType_t PRIORITY_PROCESSOR = 4;
constexpr UBaseType_t PRIORITY_COMMS     = 3;

constexpr uint32_t STACK_SENSOR    = 4096;
constexpr uint32_t STACK_PROCESSOR = 8192;
constexpr uint32_t STACK_COMMS     = 8192;

/// Core affinity: pin sensor task to core 1, comms to core 0.
constexpr BaseType_t CORE_SENSOR    = 1;
constexpr BaseType_t CORE_PROCESSOR = 1;
constexpr BaseType_t CORE_COMMS     = 0;

// ---------------------------------------------------------------------------
// Serial Debug
// ---------------------------------------------------------------------------
constexpr uint32_t SERIAL_BAUD = 115200;
