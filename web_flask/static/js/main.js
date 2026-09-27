/**
 * AutoBus AI - Autonomous Urban Bus Perception & Decision System
 * Frontend Interactions, Video Controls, and Real-Time Telemetry Client
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const video = document.getElementById("hero-bus-video");
  const videoToggleBtn = document.getElementById("video-media-toggle");
  const videoToggleIcon = document.getElementById("video-toggle-icon");
  const videoToggleText = document.getElementById("video-toggle-text");
  
  const modalOverlay = document.getElementById("arch-modal-overlay");
  const modalCloseBtn = document.getElementById("modal-close-btn");
  const navLinks = document.querySelectorAll(".nav-link");
  const btnExplore = document.getElementById("btn-explore-system");
  const btnDemoNav = document.getElementById("btn-nav-demo");
  const btnDemoHero = document.getElementById("btn-hero-demo");

  // Dynamic Telemetry Elements
  const valConfidence = document.getElementById("telemetry-confidence");
  const valSafety = document.getElementById("telemetry-safety");
  const valDecision = document.getElementById("telemetry-decision");
  const valCameras = document.getElementById("telemetry-cameras");
  const bboxPedestrian = document.querySelector(".bbox-pedestrian .bbox-tag");
  const bboxVehicle = document.querySelector(".bbox-vehicle .bbox-tag");

  // 1. Video Autoplay & Controls Handling
  if (video) {
    // Ensure video plays continuously
    video.play().catch(() => {
      // In case browser policy requires muted for autoplay
      video.muted = true;
      video.play().catch((err) => console.log("Video autoplay caught:", err));
    });

    if (videoToggleBtn) {
      videoToggleBtn.addEventListener("click", () => {
        if (video.paused) {
          video.play();
          videoToggleText.textContent = "Pause";
          videoToggleIcon.innerHTML = `<path d="M6 4h4v16H6zm8 0h4v16h-4z" fill="currentColor"/>`;
        } else {
          video.pause();
          videoToggleText.textContent = "Play";
          videoToggleIcon.innerHTML = `<path d="M8 5v14l11-7z" fill="currentColor"/>`;
        }
      });
    }
  }

  // 2. Modals & Navigation Handling
  function openModal() {
    if (modalOverlay) {
      modalOverlay.classList.add("active");
    }
  }

  function closeModal() {
    if (modalOverlay) {
      modalOverlay.classList.remove("active");
    }
  }

  if (modalCloseBtn) {
    modalCloseBtn.addEventListener("click", closeModal);
  }

  if (modalOverlay) {
    modalOverlay.addEventListener("click", (e) => {
      if (e.target === modalOverlay) closeModal();
    });
  }

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeModal();
  });

  if (btnExplore) {
    btnExplore.addEventListener("click", (e) => {
      e.preventDefault();
      openModal();
    });
  }

  navLinks.forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      navLinks.forEach(l => l.classList.remove("active"));
      link.classList.add("active");
      const target = link.dataset.target;
      if (target === "demo") {
        triggerLiveDemoTrigger();
      } else {
        openModal();
      }
    });
  });

  if (btnDemoNav) {
    btnDemoNav.addEventListener("click", (e) => {
      e.preventDefault();
      triggerLiveDemoTrigger();
    });
  }

  if (btnHeroDemo) {
    btnHeroDemo.addEventListener("click", (e) => {
      e.preventDefault();
      triggerLiveDemoTrigger();
    });
  }

  // 3. Live Demo Trigger / API Health Integration
  function triggerLiveDemoTrigger() {
    // Check if web_flask backend is reachable
    fetch("/api/health")
      .then(res => res.json())
      .then(data => {
        // Backend is online - show active notification & flash telemetry
        flashTelemetryOnline(data);
      })
      .catch(() => {
        // Standalone mode - smooth animated telemetry pulse
        simulateTelemetryPulse();
      });
  }

  function flashTelemetryOnline(data) {
    if (valDecision) {
      valDecision.textContent = "LIVE: MOVE (32 km/h)";
      valDecision.style.borderColor = "#38bdf8";
      valDecision.style.background = "rgba(56, 189, 248, 0.25)";
    }
    setTimeout(() => {
      if (valDecision) {
        valDecision.textContent = "MOVE (28 km/h)";
        valDecision.style.borderColor = "rgba(59, 130, 246, 0.5)";
        valDecision.style.background = "rgba(37, 99, 235, 0.25)";
      }
    }, 4000);
  }

  // 4. Subtle Real-Time Telemetry Micro-Fluctuations (Simulating Live Brain Updates)
  function simulateTelemetryPulse() {
    if (!valConfidence) return;
    
    // Slight realistic jitter in confidence: 91.8% to 94.6%
    const confVal = (92.4 + (Math.random() * 2.2 - 1.1)).toFixed(1);
    valConfidence.textContent = `${confVal}%`;

    // Slight bounding box confidence updates
    if (bboxPedestrian) {
      const pedConf = (0.91 + (Math.random() * 0.04)).toFixed(2);
      bboxPedestrian.textContent = `PEDESTRIAN ${pedConf}`;
    }
    if (bboxVehicle) {
      const vehConf = (0.86 + (Math.random() * 0.05)).toFixed(2);
      bboxVehicle.textContent = `LEAD VEHICLE ${vehConf} (14.2m)`;
    }
  }

  // Interval for subtle live realism
  setInterval(simulateTelemetryPulse, 2800);

  // 5. Optional WebSocket Connection to /ws/live if Socket.IO library is loaded
  if (typeof io !== "undefined") {
    try {
      const socket = io("/ws/live", {
        transports: ["websocket", "polling"],
        reconnectionAttempts: 2,
        timeout: 2000
      });

      socket.on("connect", () => {
        console.log("[AutoBus Live] Connected to live perception telemetry stream.");
        const liveIndicator = document.querySelector(".sensor-stream-indicator span.live-rec");
        if (liveIndicator) {
          liveIndicator.textContent = "LIVE 30 FPS";
          liveIndicator.style.color = "#34d399";
        }
      });

      socket.on("scene_update", (data) => {
        if (data.decision && valDecision) {
          valDecision.textContent = data.decision;
        }
        if (data.risk_level && valSafety) {
          valSafety.textContent = data.risk_level === "LOW" ? "CLEAR" : data.risk_level;
        }
      });
    } catch (e) {
      // Standalone mode graceful fallback
    }
  }
});
