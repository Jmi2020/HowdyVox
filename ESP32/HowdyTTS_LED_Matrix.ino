/*
 * HowdyTTS LED Matrix Controller
 * 
 * This sketch runs on an ESP32-S3 with an LED matrix display to show the current state
 * of the HowdyTTS voice assistant. It provides a web server with endpoints to update
 * the state of the display.
 * 
 * States:
 * - waiting: Waiting for wake word "Hey Howdy"
 * - listening: Listening to user input
 * - thinking: Processing the user's request
 * - speaking: Displaying the response text
 * - ending: Ending the conversation
 * 
 * Required libraries:
 * - WiFi
 * - WebServer
 * - MD_Parola (for LED matrix)
 * - MD_MAX72XX (dependency for MD_Parola)
 */

#include <WiFi.h>
#include <WebServer.h>
#include <MD_Parola.h>
#include <MD_MAX72xx.h>
#include <SPI.h>

// WiFi credentials
const char* ssid = "Your_WiFi_SSID";     // Your WiFi network name
const char* password = "Your_WiFi_Pass";  // Your WiFi password

// LED Matrix configuration
#define HARDWARE_TYPE MD_MAX72XX::FC16_HW
#define MAX_DEVICES 4  // Number of 8x8 modules in your matrix
#define CS_PIN 5       // CS pin for SPI

// Create instances
WebServer server(80);
MD_Parola matrix = MD_Parola(HARDWARE_TYPE, CS_PIN, MAX_DEVICES);

// Current state
String currentState = "waiting";
String scrollingText = "";

void setup() {
  Serial.begin(115200);
  
  // Initialize WiFi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println();
  Serial.print("Connected to WiFi. IP Address: ");
  Serial.println(WiFi.localIP());
  
  // Initialize LED Matrix
  matrix.begin();
  matrix.setIntensity(5);  // Set brightness (0-15)
  matrix.displayClear();
  
  // Set up server endpoints
  server.on("/state", HTTP_POST, handleStateUpdate);
  server.on("/speak", HTTP_POST, handleSpeakText);
  server.onNotFound(handleNotFound);
  
  // Start the server
  server.begin();
  Serial.println("HTTP server started");
  
  // Show initial state
  updateMatrixDisplay("waiting");
}

void loop() {
  server.handleClient();
  
  // Update the matrix animation
  if (matrix.displayAnimate()) {
    if (currentState == "speaking" && scrollingText.length() > 0) {
      // If speaking, continuously scroll the text
      matrix.displayText(scrollingText.c_str(), PA_CENTER, 50, 0, PA_SCROLL_LEFT, PA_SCROLL_LEFT);
      matrix.displayReset();
    } else {
      // For other states, display the appropriate animation
      updateMatrixDisplay(currentState);
    }
  }
}

void handleStateUpdate() {
  if (server.hasArg("state")) {
    String newState = server.arg("state");
    
    // If the state update has text for speaking
    if (newState == "speaking" && server.hasArg("text")) {
      scrollingText = server.arg("text");
      matrix.displayText(scrollingText.c_str(), PA_CENTER, 50, 0, PA_SCROLL_LEFT, PA_SCROLL_LEFT);
      matrix.displayReset();
    } else {
      updateMatrixDisplay(newState);
    }
    
    currentState = newState;
    server.send(200, "text/plain", "OK");
  } else {
    server.send(400, "text/plain", "Missing state parameter");
  }
}

void handleSpeakText() {
  if (server.hasArg("text")) {
    scrollingText = server.arg("text");
    currentState = "speaking";
    
    // Set up scrolling text
    matrix.displayText(scrollingText.c_str(), PA_CENTER, 50, 0, PA_SCROLL_LEFT, PA_SCROLL_LEFT);
    matrix.displayReset();
    
    server.send(200, "text/plain", "OK");
  } else {
    server.send(400, "text/plain", "Missing text parameter");
  }
}

void handleNotFound() {
  server.send(404, "text/plain", "Not found");
}

void updateMatrixDisplay(String state) {
  // Clear any scrolling text
  if (state != "speaking") {
    scrollingText = "";
  }
  
  // Update display based on state
  if (state == "waiting") {
    // Display a pulsing "Howdy"
    matrix.displayText("Howdy", PA_CENTER, 100, 500, PA_PRINT, PA_NO_EFFECT);
    matrix.displayReset();
  } 
  else if (state == "listening") {
    // Display "Listening..." with fade in effect
    matrix.displayText("Listening...", PA_CENTER, 100, 500, PA_FADE, PA_FADE);
    matrix.displayReset();
  } 
  else if (state == "thinking") {
    // Display "Thinking..." with scroll effect
    matrix.displayText("Thinking...", PA_LEFT, 100, 0, PA_SCROLL_RIGHT, PA_SCROLL_LEFT);
    matrix.displayReset();
  } 
  else if (state == "ending") {
    // Display "Later, partner!" with a scroll effect
    matrix.displayText("Later, partner!", PA_CENTER, 100, 2000, PA_SCROLL_UP, PA_SCROLL_DOWN);
    matrix.displayReset();
  }
  // The "speaking" state is handled in the main loop with scrolling text
}
