/**
 * Saarthi AI - Autonomous Urban Bus Perception & Decision System
 * Landing Page Controls & Navigation
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const video = document.getElementById("hero-bus-video");
  const videoToggleBtn = document.getElementById("video-media-toggle");
  const videoToggleIcon = document.getElementById("video-toggle-icon");
  const videoToggleText = document.getElementById("video-toggle-text");

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
          if (videoToggleText) videoToggleText.textContent = "Pause";
          if (videoToggleIcon) {
            videoToggleIcon.innerHTML = `<path d="M6 4h4v16H6zm8 0h4v16h-4z" fill="currentColor"/>`;
          }
        } else {
          video.pause();
          if (videoToggleText) videoToggleText.textContent = "Play";
          if (videoToggleIcon) {
            videoToggleIcon.innerHTML = `<path d="M8 5v14l11-7z" fill="currentColor"/>`;
          }
        }
      });
    }
  }

  // 2. Global helper for feature cards onclick in hero: navigate to how-it-works page
  window.openArchModal = function(phase) {
    window.location.href = "/static/how-it-works.html#technology";
  };
});
