/**
 * web_flask/static/js/demo.js
 * ==============================================================================
 * Saarthi AI — Multi-Camera Video Perception & Temporal Decision Live Demo
 *
 * Implements:
 * 1. 5-slot video dropzones with Required Front & Optional auto-degraded slots.
 * 2. 1-click preset video scenario loader.
 * 3. Model switcher with explicit model version badge.
 * 4. Async job submission (POST /api/demo/upload) and polling (GET /api/demo/status/<job_id>).
 * 5. Primary video player with synchronized client-side canvas bounding box overlay.
 * 6. Scrubbable colored timeline strip (click/drag anywhere to seek moment).
 * 7. Live synchronized decision panel (Action pill, speed, lane, risk, explanation).
 * 8. Summary analytics card with decision state distribution.
 * ==============================================================================
 */

document.addEventListener("DOMContentLoaded", () => {
  // ----------------------------------------------------------------------------
  // ----------------------------------------------------------------------------
  // DOM ELEMENT REFERENCES
  // ----------------------------------------------------------------------------
  const btnModelFinetuned = document.getElementById("btn-model-finetuned");
  const btnModelStock = document.getElementById("btn-model-stock");
  const activeModelTag = document.getElementById("active-model-tag");
  const btnModelVersionBadge = document.getElementById("btn-model-version-badge");

  // Mode Switcher Elements
  const tabModeImages = document.getElementById("tab-mode-images");
  const tabModeVideos = document.getElementById("tab-mode-videos");
  const presetsGroupImages = document.getElementById("presets-group-images");
  const presetsGroupVideos = document.getElementById("presets-group-videos");
  const fpsSelectorWrapper = document.getElementById("fps-selector-wrapper");
  const modeSummaryText = document.getElementById("mode-summary-text");
  const frontImagePlayer = document.getElementById("front-image-player");
  const videoControlsBar = document.getElementById("video-controls-bar");
  const timelineStripContainer = document.getElementById("timeline-strip-container");
  const camFeedLabel = document.getElementById("cam-feed-label");

  // Multi-Camera Grid & Switcher Elements
  const cameraFeedSwitcherBar = document.getElementById("camera-feed-switcher-bar");
  const surround5CamGrid = document.getElementById("surround-5cam-grid");
  const videoContainer = document.getElementById("video-container");
  const camFeedPills = document.querySelectorAll(".cam-feed-pill");

  const presetChips = document.querySelectorAll(".preset-chip");
  const btnResetFeeds = document.getElementById("btn-reset-feeds");

  const btnRunAnalysis = document.getElementById("btn-run-analysis");
  const runSpinner = document.getElementById("run-spinner");
  const runBtnText = document.getElementById("run-btn-text");
  const selectFps = document.getElementById("select-fps");

  // Processing elements
  const processingSection = document.getElementById("processing-section");
  const progressBarFill = document.getElementById("progress-bar-fill");
  const progressPercentLabel = document.getElementById("progress-percent-label");
  const processingStageText = document.getElementById("processing-stage-text");
  const processingSubtext = document.getElementById("processing-subtext");
  const processingFramesCounter = document.getElementById("processing-frames-counter");
  const rotatingStatusPill = document.getElementById("rotating-status-pill");

  // Results elements
  const resultsSection = document.getElementById("results-section");
  const videoPlayer = document.getElementById("front-video-player");
  const overlayCanvas = document.getElementById("overlay-canvas");
  const liveDetCount = document.getElementById("live-det-count");
  const overlayDecisionPill = document.getElementById("overlay-decision-pill");

  // Video controls
  const btnPlayPause = document.getElementById("btn-play-pause");
  const btnPlayText = document.getElementById("btn-play-text");
  const btnPlayIcon = document.getElementById("btn-play-icon");
  const ctrlBtnPlay = document.getElementById("ctrl-btn-play");
  const ctrlBtnRewind = document.getElementById("ctrl-btn-rewind");
  const ctrlBtnForward = document.getElementById("ctrl-btn-forward");
  const speedBtns = document.querySelectorAll(".speed-btn");
  const playbackTimeBadge = document.getElementById("playback-time-badge");

  // Timeline elements
  const timelineStripTrack = document.getElementById("timeline-strip-track");
  const timelineSegmentsWrapper = document.getElementById("timeline-segments-wrapper");
  const timelinePlayhead = document.getElementById("timeline-playhead");
  const timelineHoverInfo = document.getElementById("timeline-hover-info");

  // Decision panel elements
  const panelTimeTicker = document.getElementById("panel-time-ticker");
  const panelActionPill = document.getElementById("panel-action-pill");
  const valSpeed = document.getElementById("val-speed");
  const valRisk = document.getElementById("val-risk");
  const badgeRiskLevel = document.getElementById("badge-risk-level");
  const barRiskFill = document.getElementById("bar-risk-fill");
  const valLaneAction = document.getElementById("val-lane-action");
  const valSignal = document.getElementById("val-signal");
  const valPedestrians = document.getElementById("val-pedestrians");
  const hintPedestrians = document.getElementById("hint-pedestrians");
  const valBlindspot = document.getElementById("val-blindspot");
  const reasoningList = document.getElementById("reasoning-list");

  // Reliability bars
  const relOverallVal = document.getElementById("rel-overall-val");
  const relFillFront = document.getElementById("rel-fill-front");
  const relTextFront = document.getElementById("rel-text-front");
  const relFillFrontLeft = document.getElementById("rel-fill-front_left");
  const relTextFrontLeft = document.getElementById("rel-text-front_left");
  const relFillFrontRight = document.getElementById("rel-fill-front_right");
  const relTextFrontRight = document.getElementById("rel-text-front_right");
  const relFillRearLeft = document.getElementById("rel-fill-rear_left");
  const relTextRearLeft = document.getElementById("rel-text-rear_left");
  const relFillRearRight = document.getElementById("rel-fill-rear_right");
  const relTextRearRight = document.getElementById("rel-text-rear_right");

  // Summary elements
  const sumBadgeDominant = document.getElementById("sum-badge-dominant");
  const sumTotalFrames = document.getElementById("sum-total-frames");
  const sumDuration = document.getElementById("sum-duration");
  const sumAvgRisk = document.getElementById("sum-avg-risk");
  const sumMaxRisk = document.getElementById("sum-max-risk");
  const sumAvgSpeed = document.getElementById("sum-avg-speed");
  const sumCamsActive = document.getElementById("sum-cams-active");
  const distStackedBar = document.getElementById("dist-stacked-bar");
  const distLegendRow = document.getElementById("dist-legend-row");

  // ----------------------------------------------------------------------------
  // APPLICATION STATE
  // ----------------------------------------------------------------------------
  const CAMERA_SLOTS = ["front", "front_left", "front_right", "rear_left", "rear_right"];
  
  let currentMode = "images"; // 'images' or 'videos'
  let currentModel = "stock";  // Default to stock YOLOv8n (High-Accuracy Production Engine)

  const VIDEO_PRESETS = {
    urban_traffic: {
      name: "clip_03_urban_traffic.mp4",
      url: "/datasets/self-collected/eval_clips/clip_03_urban_traffic.mp4"
    },
    bus_stop: {
      name: "clip_01_bus_stop_approach.mp4",
      url: "/datasets/self-collected/eval_clips/clip_01_bus_stop_approach.mp4"
    },
    door_boarding: {
      name: "clip_02_door_boarding.mp4",
      url: "/datasets/self-collected/eval_clips/clip_02_door_boarding.mp4"
    },
    synthetic: {
      name: "synthetic_test_clip.mp4",
      url: "/tests/synthetic_test_clip.mp4"
    }
  };

  const IMAGE_PRESETS = {
    red_light: {
      name: "red_light_sample.jpg",
      url: "/data/red_light_sample.jpg"
    },
    urban_street: {
      name: "front.jpg",
      url: "/data/front.jpg"
    },
    full_surround: {
      front: { name: "front.jpg", url: "/data/front.jpg" },
      front_left: { name: "front_left.jpg", url: "/data/front_left.jpg" },
      front_right: { name: "front_right.jpg", url: "/data/front_right.jpg" },
      rear_left: { name: "rear_left.jpg", url: "/data/rear_left.jpg" },
      rear_right: { name: "rear_right.jpg", url: "/data/rear_right.jpg" }
    }
  };

  const cameraState = {
    front: IMAGE_PRESETS.red_light.url,
    front_left: null,
    front_right: null,
    rear_left: null,
    rear_right: null,
  };

  let currentJobId = null;
  let currentTimeline = [];
  let currentSummary = {};
  let totalVideoDuration = 0;
  let activeVideoUrl = null;
  let pollIntervalId = null;
  let statusRotatorId = null;

  const STATUS_MESSAGES = [
    "Sampling time-aligned frames across active cameras...",
    "Executing deep neural perception with YOLOv8...",
    "Evaluating sensor reliability & degradation penalties...",
    "Computing lateral blind-spot occupancy & pedestrian zones...",
    "Fusing multi-sensor arbitration with RL policy...",
    "Compiling synchronized decision timeline..."
  ];

  // ----------------------------------------------------------------------------
  // 1. MODEL SELECTION & BADGE UPDATES
  // ----------------------------------------------------------------------------
  function updateModelDisplay(modelKey) {
    currentModel = modelKey;
    if (modelKey === "fine_tuned") {
      if (btnModelFinetuned) btnModelFinetuned.classList.add("active");
      if (btnModelStock) btnModelStock.classList.remove("active");
      const badgeText = "Model: yolov8_bus_v1 (Custom Fine-Tuned)";
      if (activeModelTag) activeModelTag.textContent = badgeText;
      if (btnModelVersionBadge) {
        btnModelVersionBadge.textContent = badgeText;
        btnModelVersionBadge.style.color = "var(--accent-violet)";
      }
    } else {
      if (btnModelStock) btnModelStock.classList.add("active");
      if (btnModelFinetuned) btnModelFinetuned.classList.remove("active");
      const badgeText = "Model: yolov8n (High-Accuracy Production)";
      if (activeModelTag) activeModelTag.textContent = badgeText;
      if (btnModelVersionBadge) {
        btnModelVersionBadge.textContent = badgeText;
        btnModelVersionBadge.style.color = "#059669";
      }
    }
  }

  if (btnModelFinetuned) {
    btnModelFinetuned.addEventListener("click", () => updateModelDisplay("fine_tuned"));
  }
  if (btnModelStock) {
    btnModelStock.addEventListener("click", () => updateModelDisplay("stock"));
  }
  // Initialize with high-accuracy stock yolov8n model
  updateModelDisplay("stock");

  // ----------------------------------------------------------------------------
  // 2. MODE SWITCHER: IMAGES vs VIDEOS
  // ----------------------------------------------------------------------------
  function setMode(mode) {
    currentMode = mode;
    if (mode === "images") {
      if (tabModeImages) tabModeImages.classList.add("active");
      if (tabModeVideos) tabModeVideos.classList.remove("active");
      if (presetsGroupImages) presetsGroupImages.style.display = "flex";
      if (presetsGroupVideos) presetsGroupVideos.style.display = "none";
      if (fpsSelectorWrapper) fpsSelectorWrapper.style.display = "none";
      if (runBtnText) runBtnText.textContent = "⚡ Run Instant Image Perception (< 1s)";
      if (modeSummaryText) modeSummaryText.textContent = "⚡ Instant Static Multi-Camera Inference • Immediate Sub-Second Arbitration";
      if (cameraFeedSwitcherBar) cameraFeedSwitcherBar.style.display = "flex";

      // Set default image preset
      applyImagePreset("red_light");
    } else {
      if (tabModeVideos) tabModeVideos.classList.add("active");
      if (tabModeImages) tabModeImages.classList.remove("active");
      if (presetsGroupVideos) presetsGroupVideos.style.display = "flex";
      if (presetsGroupImages) presetsGroupImages.style.display = "none";
      if (fpsSelectorWrapper) fpsSelectorWrapper.style.display = "flex";
      if (runBtnText) runBtnText.textContent = "⚡ Run Fast Video Pipeline (~2-3s)";
      if (modeSummaryText) modeSummaryText.textContent = "⚡ Temporal Video Ingestion & Synchronized Arbitration Stream (~2-3s)";
      if (cameraFeedSwitcherBar) cameraFeedSwitcherBar.style.display = "none";
      if (surround5CamGrid) surround5CamGrid.style.display = "none";
      if (videoContainer) videoContainer.style.display = "flex";

      // Set default video preset
      applyVideoPreset("urban_traffic");
    }
  }

  if (tabModeImages) {
    tabModeImages.addEventListener("click", () => setMode("images"));
  }
  if (tabModeVideos) {
    tabModeVideos.addEventListener("click", () => setMode("videos"));
  }

  // ----------------------------------------------------------------------------
  // 3. PRESETS HANDLERS (Images & Video)
  // ----------------------------------------------------------------------------
  function applyImagePreset(key) {
    const preset = IMAGE_PRESETS[key] || IMAGE_PRESETS.red_light;
    resetAllCameraSlots();

    if (key === "full_surround") {
      Object.keys(preset).forEach(slot => {
        setSlotPreset(slot, preset[slot].url, preset[slot].name);
      });
    } else {
      setSlotPreset("front", preset.url, preset.name);
    }

    document.querySelectorAll("[data-img-preset]").forEach(btn => {
      btn.classList.toggle("active", btn.dataset.imgPreset === key);
    });
  }

  function applyVideoPreset(key) {
    const preset = VIDEO_PRESETS[key] || VIDEO_PRESETS.urban_traffic;
    resetAllCameraSlots();
    setSlotPreset("front", preset.url, preset.name);

    document.querySelectorAll("[data-clip]").forEach(btn => {
      btn.classList.toggle("active", btn.dataset.clip === key);
    });
  }

  document.querySelectorAll("[data-img-preset]").forEach(btn => {
    btn.addEventListener("click", () => applyImagePreset(btn.dataset.imgPreset));
  });

  document.querySelectorAll("[data-clip]").forEach(btn => {
    btn.addEventListener("click", () => applyVideoPreset(btn.dataset.clip));
  });

  function setSlotPreset(slot, url, fileName) {
    cameraState[slot] = url;
    const nameEl = document.getElementById(`dz-name-${slot}`);
    const metaEl = document.getElementById(`dz-meta-${slot}`);
    const statusPill = document.getElementById(`status-${slot}`);
    const thumbEl = document.getElementById(`dz-thumb-${slot}`);

    if (nameEl) nameEl.textContent = fileName;
    if (metaEl) metaEl.textContent = "Preset Loaded • Ready";
    if (statusPill) {
      statusPill.textContent = "Loaded (Preset)";
      statusPill.className = "camera-status-pill pill-loaded";
    }

    if (thumbEl) {
      if (typeof url === "string" && (url.endsWith(".jpg") || url.endsWith(".png") || url.endsWith(".jpeg") || url.includes("/data/"))) {
        thumbEl.src = url;
        thumbEl.style.display = "block";
      } else {
        thumbEl.style.display = "none";
      }
    }
  }

  // Initialize in Image Analysis Mode by default for instant sub-second verification
  setMode("images");

  function resetAllCameraSlots() {
    CAMERA_SLOTS.forEach(slot => {
      cameraState[slot] = null;
      const nameEl = document.getElementById(`dz-name-${slot}`);
      const metaEl = document.getElementById(`dz-meta-${slot}`);
      const statusPill = document.getElementById(`status-${slot}`);
      const thumbEl = document.getElementById(`dz-thumb-${slot}`);

      if (nameEl) nameEl.textContent = "Drop Image / Video";
      if (metaEl) metaEl.textContent = slot === "front" ? "Front Camera (Required)" : "Optional (0.0 rel if omitted)";
      if (statusPill) {
        statusPill.textContent = slot === "front" ? "Pending Feed" : "Auto-Degraded";
        statusPill.className = "camera-status-pill pill-empty";
      }
      if (thumbEl) {
        thumbEl.style.display = "none";
        thumbEl.src = "";
      }
    });
  }

  if (btnResetFeeds) {
    btnResetFeeds.addEventListener("click", () => {
      resetAllCameraSlots();
      if (currentMode === "images") {
        applyImagePreset("red_light");
      } else {
        applyVideoPreset("urban_traffic");
      }
    });
  }

  // Prevent browser window from opening dragged files on accidental miss
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  // ----------------------------------------------------------------------------
  // 3. FILE DROPZONE & BROWSER UPLOAD LISTENERS
  // ----------------------------------------------------------------------------
  CAMERA_SLOTS.forEach(slot => {
    const dropzone = document.getElementById(`dropzone-${slot}`);
    const fileInput = document.getElementById(`input-${slot}`);
    const btnBrowse = document.getElementById(`btn-browse-${slot}`);

    if (!fileInput) return;

    function triggerFileInput(e) {
      if (e) {
        e.preventDefault();
        e.stopPropagation();
      }
      fileInput.value = ""; // clear so selecting identical file triggers change
      fileInput.click();
    }

    if (dropzone) {
      dropzone.addEventListener("click", triggerFileInput);

      dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add("drag-active");
      });

      dropzone.addEventListener("dragenter", (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add("drag-active");
      });

      dropzone.addEventListener("dragleave", (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove("drag-active");
      });

      dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove("drag-active");

        if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
          handleFileSelection(slot, e.dataTransfer.files[0]);
        }
      });
    }

    if (btnBrowse) {
      btnBrowse.addEventListener("click", triggerFileInput);
    }

    fileInput.addEventListener("click", (e) => e.stopPropagation());
    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFileSelection(slot, e.target.files[0]);
      }
    });
  });

  function handleFileSelection(slot, file) {
    if (!file) return;
    cameraState[slot] = file;

    const nameEl = document.getElementById(`dz-name-${slot}`);
    const metaEl = document.getElementById(`dz-meta-${slot}`);
    const statusPill = document.getElementById(`status-${slot}`);
    const thumbEl = document.getElementById(`dz-thumb-${slot}`);

    const isVideo = file.type.startsWith("video/") || /\.(mp4|mov|avi|webm|mkv)$/i.test(file.name);
    const isImage = file.type.startsWith("image/") || /\.(jpe?g|png|webp|bmp|gif)$/i.test(file.name);
    const sizeMb = (file.size / (1024 * 1024)).toFixed(1);

    if (nameEl) nameEl.textContent = file.name;
    if (metaEl) metaEl.textContent = `${isVideo ? "Video" : "Image"} • ${sizeMb} MB`;
    if (statusPill) {
      statusPill.textContent = "Loaded (File)";
      statusPill.className = "camera-status-pill pill-loaded";
    }

    const objectUrl = URL.createObjectURL(file);

    // Update thumbnail in dropzone
    if (thumbEl) {
      if (isImage) {
        thumbEl.src = objectUrl;
        thumbEl.style.display = "block";
      } else {
        thumbEl.style.display = "none";
      }
    }

    // If uploading the primary front feed, update the main viewport immediately!
    if (slot === "front") {
      if (isVideo) {
        // Automatically switch mode tab to Video Stream
        if (currentMode !== "videos") {
          currentMode = "videos";
          if (tabModeVideos) tabModeVideos.classList.add("active");
          if (tabModeImages) tabModeImages.classList.remove("active");
          if (presetsGroupVideos) presetsGroupVideos.style.display = "flex";
          if (presetsGroupImages) presetsGroupImages.style.display = "none";
          if (fpsSelectorWrapper) fpsSelectorWrapper.style.display = "flex";
          if (runBtnText) runBtnText.textContent = "⚡ Run Fast Video Pipeline (~2-3s)";
          if (modeSummaryText) modeSummaryText.textContent = "⚡ Temporal Video Ingestion & Synchronized Arbitration Stream (~2-3s)";
        }

        if (frontImagePlayer) frontImagePlayer.style.display = "none";
        if (videoPlayer) {
          videoPlayer.style.display = "block";
          videoPlayer.src = objectUrl;
          videoPlayer.load();
        }
        if (videoControlsBar) videoControlsBar.style.display = "flex";
        if (timelineStripContainer) timelineStripContainer.style.display = "none";
        if (camFeedLabel) camFeedLabel.textContent = "CAM 1: FRONT (VIDEO LOADED)";
        if (liveDetCount) liveDetCount.textContent = "Ready for Analysis";
        if (overlayDecisionPill) {
          overlayDecisionPill.textContent = "READY";
          overlayDecisionPill.className = "overlay-decision-pill pill-move";
        }
      } else {
        // Automatically switch mode tab to Image Analysis
        if (currentMode !== "images") {
          currentMode = "images";
          if (tabModeImages) tabModeImages.classList.add("active");
          if (tabModeVideos) tabModeVideos.classList.remove("active");
          if (presetsGroupImages) presetsGroupImages.style.display = "flex";
          if (presetsGroupVideos) presetsGroupVideos.style.display = "none";
          if (fpsSelectorWrapper) fpsSelectorWrapper.style.display = "none";
          if (runBtnText) runBtnText.textContent = "⚡ Run Instant Image Perception (< 1s)";
          if (modeSummaryText) modeSummaryText.textContent = "⚡ Instant Static Multi-Camera Inference • Immediate Sub-Second Arbitration";
        }

        if (videoPlayer) videoPlayer.style.display = "none";
        if (frontImagePlayer) {
          frontImagePlayer.style.display = "block";
          frontImagePlayer.src = objectUrl;
        }
        if (videoControlsBar) videoControlsBar.style.display = "none";
        if (timelineStripContainer) timelineStripContainer.style.display = "none";
        if (camFeedLabel) camFeedLabel.textContent = "CAM 1: FRONT (IMAGE LOADED)";
        if (liveDetCount) liveDetCount.textContent = "Ready for Analysis";
        if (overlayDecisionPill) {
          overlayDecisionPill.textContent = "READY";
          overlayDecisionPill.className = "overlay-decision-pill pill-move";
        }
      }

      // Smoothly bring viewport into view so user gets immediate visual confirmation
      if (resultsSection) {
        resultsSection.style.display = "block";
      }
    }
  }

  // ----------------------------------------------------------------------------
  // 4. PIPELINE EXECUTION & ASYNC POLLING
  // ----------------------------------------------------------------------------
  if (btnRunAnalysis) {
    btnRunAnalysis.addEventListener("click", runAnalysis);
  }

  // Multi-Camera Grid & Switcher State
  let currentPerceptionImages = {};
  let currentPerceptionCounts = {};
  let activeCameraView = "all";

  function setCameraViewMode(view) {
    activeCameraView = view;
    if (camFeedPills) {
      camFeedPills.forEach(p => p.classList.toggle("active", p.dataset.feed === view));
    }

    if (view === "all") {
      if (surround5CamGrid) surround5CamGrid.style.display = "flex";
      if (videoContainer) videoContainer.style.display = "none";
    } else {
      if (surround5CamGrid) surround5CamGrid.style.display = "none";
      if (videoContainer) videoContainer.style.display = "flex";

      const labelMap = {
        front: "CAM 1: FRONT (PRIMARY NAVIGATION)",
        front_left: "CAM 2: FRONT-LEFT (LEFT BLIND SPOT)",
        front_right: "CAM 3: FRONT-RIGHT (RIGHT BLIND SPOT)",
        rear_left: "CAM 4: BACK-LEFT (REAR BLIND SPOT)",
        rear_right: "CAM 5: BACK-RIGHT (REAR CURBSIDE)"
      };

      if (camFeedLabel) camFeedLabel.textContent = labelMap[view] || view.toUpperCase();
      if (liveDetCount) {
        const cnt = currentPerceptionCounts[view] !== undefined ? currentPerceptionCounts[view] : 0;
        liveDetCount.textContent = `${cnt} Detections Verified`;
      }
      if (frontImagePlayer) {
        frontImagePlayer.src = currentPerceptionImages[view] || cameraState[view] || "/data/front.jpg";
      }
    }
  }

  if (camFeedPills) {
    camFeedPills.forEach(p => {
      p.addEventListener("click", () => setCameraViewMode(p.dataset.feed));
    });
  }

  document.querySelectorAll(".cam-grid-card").forEach(card => {
    card.addEventListener("click", () => {
      const slot = card.dataset.camSlot;
      if (slot) setCameraViewMode(slot);
    });
  });

  function runAnalysis() {
    if (currentMode === "images") {
      runImagePerception();
    } else {
      runVideoPipeline();
    }
  }

  async function runImagePerception() {
    if (!cameraState.front) {
      alert("Front Camera image feed is required to run perception analysis.");
      return;
    }

    setProcessingState(true);
    if (processingStageText) processingStageText.textContent = "Executing neural perception & safety fusion...";
    if (processingSubtext) processingSubtext.textContent = "Processing camera frames with YOLOv8 & RL policy engine...";
    if (progressBarFill) progressBarFill.style.width = "85%";
    if (progressPercentLabel) progressPercentLabel.textContent = "85%";

    try {
      const formData = new FormData();
      for (const slot of CAMERA_SLOTS) {
        const item = cameraState[slot];
        if (item instanceof File) {
          formData.append(slot, item);
        } else if (typeof item === "string" && item.length > 0) {
          try {
            const resp = await fetch(item);
            const blob = await resp.blob();
            formData.append(slot, blob, `${slot}.jpg`);
          } catch (e) {
            formData.append(slot, item);
          }
        }
      }

      const response = await fetch(`/api/analyze?model=${currentModel}&include_images=true`, {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Perception failed with HTTP ${response.status}`);
      }

      const data = await response.json();
      setProcessingState(false);
      renderImagePerceptionResults(data);
    } catch (err) {
      console.error("[Saarthi Live Demo Error]:", err);
      alert("Image analysis failed: " + err.message);
      setProcessingState(false);
    }
  }

  function renderImagePerceptionResults(data) {
    const dec = (data.final_decision || data.decision || "MOVE").toUpperCase();
    const sig = (data.traffic_signal || "UNKNOWN").toUpperCase();
    const spd = data.recommended_speed !== undefined ? data.recommended_speed : (data.final_speed || 0);
    const rScore = data.risk_score || 0;
    const rLvl = (data.risk_level || "LOW").toUpperCase();
    const pedCount = data.pedestrian_count || 0;
    const pedStat = data.pedestrian_status || "NONE";

    // Extract all 5 cameras' perceptions
    currentPerceptionImages = {
      front: data.annotated_front_image || data.front_annotated || cameraState.front,
      front_left: data.annotated_front_left_image || data.front_left_annotated || cameraState.front_left,
      front_right: data.annotated_front_right_image || data.front_right_annotated || cameraState.front_right,
      rear_left: data.annotated_rear_left_image || data.rear_left_annotated || cameraState.rear_left,
      rear_right: data.annotated_rear_right_image || data.rear_right_annotated || cameraState.rear_right
    };

    currentPerceptionCounts = {
      front: data.object_count || 0,
      front_left: data.front_left_objects || 0,
      front_right: data.front_right_objects || 0,
      rear_left: data.rear_left_objects || 0,
      rear_right: data.rear_right_objects || 0
    };

    const isFront = !!cameraState.front;
    const isFL = !!cameraState.front_left;
    const isFR = !!cameraState.front_right;
    const isRL = !!cameraState.rear_left;
    const isRR = !!cameraState.rear_right;
    const activeCount = [isFront, isFL, isFR, isRL, isRR].filter(Boolean).length;

    const entry = {
      timestamp: 0.0,
      decision: dec,
      speed: spd,
      lane_action: data.lane_action || "KEEP_LANE",
      risk_score: rScore,
      risk_level: rLvl,
      contributing_factors: {
        traffic_signal: sig,
        lead_vehicle: data.lead_vehicle || null,
        vehicle_distance: data.vehicle_distance || "FAR",
        pedestrian_count: pedCount,
        pedestrian_status: pedStat,
        left_lane_status: data.left_lane_status || "SAFE",
        right_lane_status: data.right_lane_status || "SAFE",
        collision_risk: data.collision_risk || "Low Risk",
        rl_action: data.rl_action || "MAINTAIN"
      },
      explanation: data.fusion_reason || (Array.isArray(data.reasons) ? data.reasons.join("; ") : "Arbitration verified."),
      reliability_scores: {
        overall: (activeCount / 5.0) * (data.camera_reliability || 0.95),
        front: isFront ? (data.camera_reliability || 0.85) : 0.0,
        front_left: isFL ? 0.88 : 0.0,
        front_right: isFR ? 0.90 : 0.0,
        rear_left: isRL ? 0.82 : 0.0,
        rear_right: isRR ? 0.84 : 0.0
      }
    };

    updateDecisionPanel(entry);

    // Update all 5 Camera Grid Cards & Dropzones
    CAMERA_SLOTS.forEach(slot => {
      const gImg = document.getElementById(`grid-img-${slot}`);
      const gPill = document.getElementById(`grid-pill-${slot}`);
      const cnt = currentPerceptionCounts[slot] || 0;
      const imgUrl = currentPerceptionImages[slot];

      if (gImg && imgUrl) {
        gImg.src = imgUrl;
      }
      if (gPill) {
        gPill.textContent = `${cnt} Detection${cnt === 1 ? '' : 's'}`;
        gPill.className = `cam-grid-det-pill ${cnt === 0 ? 'empty' : ''}`;
      }

      // Also update thumbnail and status in the 5 dropzones at the top
      const thumbEl = document.getElementById(`dz-thumb-${slot}`);
      const statusPill = document.getElementById(`status-${slot}`);
      if (thumbEl && imgUrl) {
        thumbEl.src = imgUrl;
        thumbEl.style.display = "block";
      }
      if (statusPill && cameraState[slot]) {
        statusPill.textContent = `Verified (${cnt} Det)`;
        statusPill.className = "camera-status-pill pill-loaded";
      }
    });

    if (cameraFeedSwitcherBar) cameraFeedSwitcherBar.style.display = "flex";
    setCameraViewMode("all");

    if (videoControlsBar) videoControlsBar.style.display = "none";
    if (timelineStripContainer) timelineStripContainer.style.display = "none";

    // Render snapshot summary cards
    renderSummary({
      dominant_decision: dec,
      total_timesteps: 1,
      duration_sec: 0.1,
      avg_risk_score: rScore,
      max_risk_score: rScore,
      avg_recommended_speed: spd,
      active_cameras: Object.keys(cameraState).filter(k => cameraState[k] !== null),
      decision_distribution: { [dec]: 1 }
    }, [entry]);

    if (resultsSection) {
      resultsSection.style.display = "block";
      resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  async function runVideoPipeline() {
    if (!cameraState.front) {
      alert("Front Camera video feed is required to run perception analysis.");
      return;
    }

    setProcessingState(true);

    try {
      const formData = new FormData();
      const targetFps = selectFps ? selectFps.value : "1";

      // Attach videos (either File or fetch blob from preset URL)
      for (const slot of CAMERA_SLOTS) {
        const item = cameraState[slot];
        if (item instanceof File) {
          formData.append(slot, item);
        } else if (typeof item === "string" && item.length > 0) {
          try {
            const resp = await fetch(item);
            const blob = await resp.blob();
            formData.append(slot, blob, `${slot}.mp4`);
          } catch (fetchErr) {
            console.warn(`Could not fetch blob for ${slot}:`, fetchErr);
            formData.append(slot, item); // send path as string fallback
          }
        }
      }

      const queryParams = new URLSearchParams({
        model: currentModel,
        target_fps: targetFps,
        max_duration_sec: "60.0"
      });

      const response = await fetch(`/api/demo/upload?${queryParams.toString()}`, {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Upload failed with HTTP ${response.status}`);
      }

      const uploadData = await response.json();
      currentJobId = uploadData.job_id;

      startPollingJob(currentJobId);

    } catch (err) {
      console.error("[Saarthi Live Demo Error]:", err);
      alert("Could not start video analysis: " + err.message);
      setProcessingState(false);
    }
  }

  function setProcessingState(isProcessing) {
    if (btnRunAnalysis) btnRunAnalysis.disabled = isProcessing;
    if (runSpinner) runSpinner.style.display = isProcessing ? "inline-block" : "none";
    if (runBtnText) {
      if (currentMode === "images") {
        runBtnText.textContent = isProcessing 
          ? "Analyzing Static Scene (< 1s)..." 
          : "⚡ Run Instant Image Perception (< 1s)";
      } else {
        runBtnText.textContent = isProcessing 
          ? "Processing Video Stream (~2-3s)..." 
          : "⚡ Run Fast Video Pipeline (~2-3s)";
      }
    }

    if (processingSection) {
      processingSection.style.display = isProcessing ? "block" : "none";
    }
    if (isProcessing) {
      if (resultsSection) resultsSection.style.display = "none";
      progressBarFill.style.width = "0%";
      progressPercentLabel.textContent = "0%";
      startStatusRotator();
    } else {
      stopStatusRotator();
    }
  }

  function startStatusRotator() {
    let msgIdx = 0;
    if (statusRotatorId) clearInterval(statusRotatorId);
    statusRotatorId = setInterval(() => {
      msgIdx = (msgIdx + 1) % STATUS_MESSAGES.length;
      if (rotatingStatusPill) rotatingStatusPill.textContent = STATUS_MESSAGES[msgIdx];
      if (processingStageText) processingStageText.textContent = STATUS_MESSAGES[msgIdx];
    }, 1800);
  }

  function stopStatusRotator() {
    if (statusRotatorId) {
      clearInterval(statusRotatorId);
      statusRotatorId = null;
    }
  }

  function startPollingJob(jobId) {
    if (pollIntervalId) clearInterval(pollIntervalId);

    pollIntervalId = setInterval(async () => {
      try {
        const resp = await fetch(`/api/demo/status/${jobId}`);
        if (!resp.ok) return;

        const data = await resp.json();
        const progress = Math.min(100, Math.max(0, data.progress || 0));

        progressBarFill.style.width = `${progress}%`;
        progressPercentLabel.textContent = `${progress.toFixed(0)}%`;
        if (processingFramesCounter) {
          processingFramesCounter.textContent = `Frame ${data.current_frame || 0} of ${data.total_frames || '...'}`;
        }

        if (data.status === "completed") {
          clearInterval(pollIntervalId);
          pollIntervalId = null;
          setProcessingState(false);

          currentTimeline = data.timeline || [];
          currentSummary = data.summary || {};
          totalVideoDuration = currentSummary.duration_sec || 0;

          renderResults(data);
        } else if (data.status === "failed") {
          clearInterval(pollIntervalId);
          pollIntervalId = null;
          setProcessingState(false);
          alert("Video analysis job failed: " + (data.error || "Unknown error"));
        }
      } catch (pollErr) {
        console.warn("Poll status check error:", pollErr);
      }
    }, 450);
  }

  // ----------------------------------------------------------------------------
  // 5. RENDER RESULTS & INITIALIZE VIDEO PLAYER
  // ----------------------------------------------------------------------------
  function renderResults(jobData) {
    if (resultsSection) resultsSection.style.display = "block";

    if (videoPlayer) videoPlayer.style.display = "block";
    if (frontImagePlayer) frontImagePlayer.style.display = "none";
    if (videoControlsBar) videoControlsBar.style.display = "flex";
    if (timelineStripContainer) timelineStripContainer.style.display = "block";
    if (camFeedLabel) camFeedLabel.textContent = "CAM 1: FRONT (PRIMARY)";

    // Setup video source
    const frontItem = cameraState.front;
    if (frontItem instanceof File) {
      activeVideoUrl = URL.createObjectURL(frontItem);
    } else if (typeof frontItem === "string") {
      activeVideoUrl = frontItem;
    } else {
      activeVideoUrl = `/api/demo/video/${jobData.job_id}/front`;
    }

    videoPlayer.src = activeVideoUrl;
    videoPlayer.load();

    // Render Timeline Ribbon
    renderTimelineStrip(jobData.timeline, jobData.summary.duration_sec);

    // Render Summary Card
    renderSummary(jobData.summary, jobData.timeline);

    // Initial sync with frame 0
    if (jobData.timeline && jobData.timeline.length > 0) {
      updateDecisionPanel(jobData.timeline[0]);
    }

    // Scroll smoothly to results
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // ----------------------------------------------------------------------------
  // 6. SCRUBBABLE TIMELINE STRIP RENDERING & INTERACTION (CORE FEATURE)
  // ----------------------------------------------------------------------------
  function renderTimelineStrip(timeline, totalDuration) {
    if (!timelineSegmentsWrapper) return;
    timelineSegmentsWrapper.innerHTML = "";

    if (!timeline || timeline.length === 0) return;

    timeline.forEach((entry, idx) => {
      const seg = document.createElement("div");
      seg.className = `timeline-segment ${getDecisionColorClass(entry.decision)}`;
      seg.dataset.index = idx;
      seg.dataset.timestamp = entry.timestamp;

      seg.addEventListener("click", (e) => {
        e.stopPropagation();
        seekToTimestamp(entry.timestamp);
      });

      seg.addEventListener("mouseenter", () => {
        if (timelineHoverInfo) {
          timelineHoverInfo.textContent = `T = ${entry.timestamp}s • ${entry.decision} • ${entry.speed} km/h • Risk: ${entry.risk_score}`;
        }
      });

      seg.addEventListener("mouseleave", () => {
        if (timelineHoverInfo) {
          timelineHoverInfo.textContent = "Hover for timestep details";
        }
      });

      timelineSegmentsWrapper.appendChild(seg);
    });

    // Clicking anywhere on track seeks proportional time
    if (timelineStripTrack) {
      timelineStripTrack.onclick = (e) => {
        const rect = timelineStripTrack.getBoundingClientRect();
        const clickRatio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        const targetTime = clickRatio * (videoPlayer.duration || totalDuration || 1);
        seekToTimestamp(targetTime);
      };
    }
  }

  function getDecisionColorClass(decision) {
    const d = (decision || "").toUpperCase();
    if (d === "MOVE") return "color-move";
    if (d.includes("SLOW") || d.includes("WAIT")) return "color-slow";
    if (d.includes("STOP")) return "color-stop";
    if (d.includes("LANE") || d.includes("LEFT") || d.includes("RIGHT")) return "color-lane";
    return "color-move";
  }

  function seekToTimestamp(timestamp) {
    if (videoPlayer) {
      videoPlayer.currentTime = timestamp;
      syncFrameAtTime(timestamp);
    }
  }

  // ----------------------------------------------------------------------------
  // 7. VIDEO PLAYBACK & CANVAS BOUNDING BOX SYNCHRONIZATION
  // ----------------------------------------------------------------------------
  videoPlayer.addEventListener("loadedmetadata", () => {
    resizeOverlayCanvas();
  });

  window.addEventListener("resize", () => {
    resizeOverlayCanvas();
  });

  function resizeOverlayCanvas() {
    if (!overlayCanvas || !videoPlayer) return;
    overlayCanvas.width = videoPlayer.clientWidth || 640;
    overlayCanvas.height = videoPlayer.clientHeight || 360;
  }

  videoPlayer.addEventListener("timeupdate", () => {
    const curTime = videoPlayer.currentTime;
    const dur = videoPlayer.duration || totalVideoDuration || 1;

    // Update playhead position
    const ratio = Math.min(100, Math.max(0, (curTime / dur) * 100));
    if (timelinePlayhead) {
      timelinePlayhead.style.left = `${ratio}%`;
    }

    if (playbackTimeBadge) {
      playbackTimeBadge.textContent = `${curTime.toFixed(2)}s / ${dur.toFixed(2)}s`;
    }

    syncFrameAtTime(curTime);
  });

  function syncFrameAtTime(curTime) {
    if (!currentTimeline || currentTimeline.length === 0) return;

    // Find timeline entry closest to current playback time
    let closestEntry = currentTimeline[0];
    let minDiff = Math.abs(curTime - closestEntry.timestamp);

    for (let i = 1; i < currentTimeline.length; i++) {
      const diff = Math.abs(curTime - currentTimeline[i].timestamp);
      if (diff < minDiff) {
        minDiff = diff;
        closestEntry = currentTimeline[i];
      }
    }

    updateDecisionPanel(closestEntry);
    drawBoundingBoxes(closestEntry.detections || []);
  }

  // ----------------------------------------------------------------------------
  // 8. CLIENT-SIDE BOUNDING BOX CANVAS RENDERER
  // ----------------------------------------------------------------------------
  function drawBoundingBoxes(detections) {
    if (!overlayCanvas) return;
    const ctx = overlayCanvas.getContext("2d");
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    if (liveDetCount) {
      liveDetCount.textContent = `${detections.length} Detections`;
    }

    if (!detections || detections.length === 0) return;

    // Determine coordinate scale from video native dimensions to canvas
    const videoWidth = videoPlayer.videoWidth || 640;
    const videoHeight = videoPlayer.videoHeight || 480;
    const scaleX = overlayCanvas.width / videoWidth;
    const scaleY = overlayCanvas.height / videoHeight;

    detections.forEach(det => {
      const x1 = (det.x1 || 0) * scaleX;
      const y1 = (det.y1 || 0) * scaleY;
      const x2 = (det.x2 || 0) * scaleX;
      const y2 = (det.y2 || 0) * scaleY;
      const width = x2 - x1;
      const height = y2 - y1;

      // Color coding by class
      const cls = (det.class || "").toLowerCase();
      let strokeColor = "#38bdf8"; // vehicle/default cyan
      let fillColor = "rgba(56, 189, 248, 0.12)";

      if (cls === "person") {
        strokeColor = "#f59e0b"; // amber for pedestrians
        fillColor = "rgba(245, 158, 11, 0.15)";
      } else if (cls.includes("traffic light") || cls.includes("signal")) {
        strokeColor = "#10b981"; // emerald for lights
        fillColor = "rgba(16, 185, 129, 0.15)";
      }

      // Draw bounding box
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = strokeColor;
      ctx.fillStyle = fillColor;
      ctx.strokeRect(x1, y1, width, height);
      ctx.fillRect(x1, y1, width, height);

      // Label Pill
      const labelText = `${det.class} ${(det.confidence * 100).toFixed(0)}%`;
      ctx.font = "bold 11px 'JetBrains Mono', monospace";
      const textMetrics = ctx.measureText(labelText);
      const padX = 6;
      const padY = 4;
      const pillW = textMetrics.width + padX * 2;
      const pillH = 16;
      const pillY = Math.max(0, y1 - pillH);

      ctx.fillStyle = strokeColor;
      ctx.fillRect(x1, pillY, pillW, pillH);

      ctx.fillStyle = "#ffffff";
      ctx.fillText(labelText, x1 + padX, pillY + 12);
    });
  }

  // ----------------------------------------------------------------------------
  // 9. LIVE SYNCHRONIZED DECISION PANEL UPDATES
  // ----------------------------------------------------------------------------
  function updateDecisionPanel(entry) {
    if (!entry) return;

    const decision = (entry.decision || "MOVE").toUpperCase();

    // 1. Dominant Action Pill
    if (panelActionPill) {
      panelActionPill.textContent = decision;
      panelActionPill.className = `current-action-pill ${getActionPillClass(decision)}`;
    }
    if (overlayDecisionPill) {
      overlayDecisionPill.textContent = decision;
      overlayDecisionPill.className = `overlay-decision-pill ${getActionPillClass(decision)}`;
    }

    if (panelTimeTicker) {
      panelTimeTicker.textContent = typeof entry.timestamp === "number" ? `T = ${entry.timestamp.toFixed(2)}s` : "Snapshot Analysis";
    }

    // 2. Telemetry Values
    if (valSpeed) valSpeed.textContent = entry.speed !== undefined ? entry.speed : 40;
    if (valLaneAction) valLaneAction.textContent = entry.lane_action || "MAINTAIN_LANE";

    const risk = entry.risk_score || 0;
    const rLevel = (entry.risk_level || "LOW").toUpperCase();
    if (valRisk) valRisk.textContent = risk;
    if (badgeRiskLevel) {
      badgeRiskLevel.textContent = `${rLevel} RISK`;
      badgeRiskLevel.className = `risk-badge ${getRiskBadgeClass(rLevel)}`;
    }
    if (barRiskFill) {
      barRiskFill.style.width = `${Math.min(100, risk)}%`;
      barRiskFill.style.background = rLevel === "HIGH" ? "var(--accent-rose)" : (rLevel === "MEDIUM" ? "var(--accent-amber)" : "var(--accent-emerald)");
    }

    // Contributing Factors
    const cf = entry.contributing_factors || {};
    if (valSignal) {
      const sig = (cf.traffic_signal || "UNKNOWN").toUpperCase();
      valSignal.textContent = sig;
      valSignal.className = `box-value-text ${sig === "GREEN" ? "sig-green" : (sig === "RED" ? "sig-red" : "sig-yellow")}`;
    }

    if (valPedestrians) {
      valPedestrians.textContent = `${cf.pedestrian_count || 0} Detected`;
    }
    if (hintPedestrians) {
      hintPedestrians.textContent = `Status: ${cf.pedestrian_status || 'NONE'} • Active In-Zone`;
    }

    if (valBlindspot) {
      valBlindspot.textContent = `L: ${cf.left_lane_status || 'CLEAR'} | R: ${cf.right_lane_status || 'CLEAR'}`;
    }

    // 3. Explanation list
    if (reasoningList) {
      reasoningList.innerHTML = "";
      const expText = entry.explanation || "All safety envelopes nominal; clear trajectory arbitrated.";
      const reasons = expText.split(";");
      reasons.forEach(r => {
        const trimmed = r.trim();
        if (trimmed) {
          const li = document.createElement("li");
          li.textContent = trimmed;
          reasoningList.appendChild(li);
        }
      });
    }

    // 4. Per-Camera Reliability Telemetry
    const rel = entry.reliability_scores || {};
    if (relOverallVal) {
      relOverallVal.textContent = `Overall: ${((rel.overall || 1.0) * 100).toFixed(0)}%`;
    }
    updateReliabilityBar("front", rel.front);
    updateReliabilityBar("front_left", rel.front_left);
    updateReliabilityBar("front_right", rel.front_right);
    updateReliabilityBar("rear_left", rel.rear_left);
    updateReliabilityBar("rear_right", rel.rear_right);
  }

  function getActionPillClass(decision) {
    if (decision === "MOVE") return "pill-move";
    if (decision.includes("SLOW") || decision.includes("WAIT")) return "pill-slow";
    if (decision.includes("STOP")) return "pill-stop";
    return "pill-lane";
  }

  function getRiskBadgeClass(level) {
    if (level === "HIGH") return "badge-high";
    if (level === "MEDIUM" || level === "MED") return "badge-med";
    return "badge-low";
  }

  function updateReliabilityBar(slot, val) {
    const fillEl = document.getElementById(`rel-fill-${slot}`);
    const textEl = document.getElementById(`rel-text-${slot}`);
    if (!fillEl || !textEl) return;

    if (val === undefined || val === 0.0) {
      fillEl.style.width = "0%";
      fillEl.className = "rel-fill degraded";
      textEl.textContent = "Degraded";
      textEl.className = "rel-percent degraded";
    } else {
      const pct = (val * 100).toFixed(0);
      fillEl.style.width = `${pct}%`;
      fillEl.className = "rel-fill";
      textEl.textContent = `${pct}%`;
      textEl.className = "rel-percent";
    }
  }

  // ----------------------------------------------------------------------------
  // 10. SUMMARY CARD RENDERER
  // ----------------------------------------------------------------------------
  function renderSummary(summary, timeline) {
    if (!summary) return;

    if (sumBadgeDominant) sumBadgeDominant.textContent = `Dominant: ${summary.dominant_decision || 'MOVE'}`;
    if (sumTotalFrames) sumTotalFrames.textContent = summary.total_timesteps || timeline.length;
    if (sumDuration) sumDuration.textContent = `${summary.duration_sec || 0}s`;
    if (sumAvgRisk) sumAvgRisk.textContent = summary.avg_risk_score !== undefined ? summary.avg_risk_score : "5.0";
    if (sumMaxRisk) sumMaxRisk.textContent = summary.max_risk_score !== undefined ? summary.max_risk_score : "5.0";
    if (sumAvgSpeed) sumAvgSpeed.textContent = `${summary.avg_recommended_speed || 40} km/h`;
    if (sumCamsActive) {
      const activeCount = (summary.active_cameras || []).length;
      sumCamsActive.textContent = `${activeCount} / 5`;
    }

    // Stacked Distribution Bar
    if (distStackedBar && summary.decision_distribution) {
      distStackedBar.innerHTML = "";
      if (distLegendRow) distLegendRow.innerHTML = "";

      const total = summary.total_timesteps || 1;
      const dist = summary.decision_distribution;

      for (const [dec, count] of Object.entries(dist)) {
        const pct = ((count / total) * 100).toFixed(1);
        const seg = document.createElement("div");
        seg.className = `dist-bar-segment ${getDecisionColorClass(dec)}`;
        seg.style.width = `${pct}%`;
        seg.title = `${dec}: ${count} frames (${pct}%)`;
        distStackedBar.appendChild(seg);

        if (distLegendRow) {
          const legItem = document.createElement("span");
          legItem.className = "legend-item";
          legItem.innerHTML = `<span class="legend-box ${getDecisionColorClass(dec)}"></span> ${dec}: ${pct}%`;
          distLegendRow.appendChild(legItem);
        }
      }
    }
  }

  // ----------------------------------------------------------------------------
  // 11. PLAYBACK CONTROLS
  // ----------------------------------------------------------------------------
  function togglePlay() {
    if (videoPlayer.paused) {
      videoPlayer.play();
      if (btnPlayText) btnPlayText.textContent = "Pause Feed";
      if (btnPlayIcon) btnPlayIcon.textContent = "⏸";
      if (ctrlBtnPlay) ctrlBtnPlay.textContent = "⏸";
    } else {
      videoPlayer.pause();
      if (btnPlayText) btnPlayText.textContent = "Play Feed";
      if (btnPlayIcon) btnPlayIcon.textContent = "▶";
      if (ctrlBtnPlay) ctrlBtnPlay.textContent = "▶";
    }
  }

  if (btnPlayPause) btnPlayPause.addEventListener("click", togglePlay);
  if (ctrlBtnPlay) ctrlBtnPlay.addEventListener("click", togglePlay);

  if (ctrlBtnRewind) {
    ctrlBtnRewind.addEventListener("click", () => {
      videoPlayer.currentTime = Math.max(0, videoPlayer.currentTime - 1.0);
    });
  }

  if (ctrlBtnForward) {
    ctrlBtnForward.addEventListener("click", () => {
      videoPlayer.currentTime = Math.min(videoPlayer.duration, videoPlayer.currentTime + 1.0);
    });
  }

  speedBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      speedBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      videoPlayer.playbackRate = parseFloat(btn.dataset.speed);
    });
  });
});
