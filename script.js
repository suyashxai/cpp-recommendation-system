/**
 * static/js/script.js
 * Client-side form validation and UI helpers
 */

"use strict";

// ── Mobile nav toggle ─────────────────────────────────────────
document.addEventListener("DOMContentLoaded", function () {
  const toggle = document.getElementById("navToggle");
  const navLinks = document.getElementById("navLinks");

  if (toggle && navLinks) {
    toggle.addEventListener("click", function () {
      navLinks.classList.toggle("open");
    });
  }

  // ── Client-side form validation ───────────────────────────────
  const form = document.getElementById("predForm");
  if (form) {
    form.addEventListener("submit", function (e) {
      let valid = true;

      // Remove previous error styles
      form.querySelectorAll("input").forEach(function (input) {
        input.classList.remove("input-error");
      });

      const rules = {
        N:           { min: 0,    max: 140,  label: "Nitrogen (N)" },
        P:           { min: 5,    max: 145,  label: "Phosphorus (P)" },
        K:           { min: 5,    max: 205,  label: "Potassium (K)" },
        temperature: { min: 0,    max: 50,   label: "Temperature" },
        humidity:    { min: 0,    max: 100,  label: "Humidity" },
        ph:          { min: 0,    max: 14,   label: "Soil pH" },
        rainfall:    { min: 0,    max: 300,  label: "Rainfall" },
      };

      const errors = [];

      Object.keys(rules).forEach(function (field) {
        const input = document.getElementById(field);
        if (!input) return;

        const raw   = input.value.trim();
        const rule  = rules[field];
        const value = parseFloat(raw);

        if (raw === "" || isNaN(value)) {
          input.classList.add("input-error");
          errors.push(rule.label + " is required and must be a number.");
          valid = false;
          return;
        }

        if (value < rule.min || value > rule.max) {
          input.classList.add("input-error");
          errors.push(
            rule.label + " must be between " + rule.min + " and " + rule.max + "."
          );
          valid = false;
        }
      });

      if (!valid) {
        e.preventDefault();

        // Display errors at the top of the form
        let alertBox = document.getElementById("jsErrors");
        if (!alertBox) {
          alertBox = document.createElement("div");
          alertBox.id = "jsErrors";
          alertBox.className = "alert alert-error";
          form.parentNode.insertBefore(alertBox, form);
        }
        alertBox.innerHTML =
          "<strong>⚠ Please fix the following errors:</strong><ul>" +
          errors.map(function (e) { return "<li>" + e + "</li>"; }).join("") +
          "</ul>";
        alertBox.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    });

    // Clear error on input
    form.querySelectorAll("input").forEach(function (input) {
      input.addEventListener("input", function () {
        input.classList.remove("input-error");
        const alertBox = document.getElementById("jsErrors");
        if (alertBox) alertBox.remove();
      });
    });
  }

  // ── Animate confidence bars on result page ───────────────────
  document.querySelectorAll(".confidence-bar-inner, .prob-bar-inner").forEach(function (bar) {
    const targetWidth = bar.style.width;
    bar.style.width = "0%";
    setTimeout(function () {
      bar.style.width = targetWidth;
    }, 150);
  });
});
