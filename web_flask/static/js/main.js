/**
 * AutoBus AI - Autonomous Urban Bus Perception & Decision System
 * Frontend Interactions and Video Controls
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

  // 1. Video Autoplay & Controls Handling
  if (video) {
    video.play().catch(() => {
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
      openModal();
    });
  });

  if (btnDemoNav) {
    btnDemoNav.addEventListener("click", (e) => {
      e.preventDefault();
      openModal();
    });
  }

  if (btnHeroDemo) {
    btnHeroDemo.addEventListener("click", (e) => {
      e.preventDefault();
      openModal();
    });
  }
});
