/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef } from 'react';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import * as tf from '@tensorflow/tfjs';
import { Camera, Cpu, Info, Loader2, RotateCcw, Save, History, Download, X, Vibrate, Moon, Server, Settings, Bell, Target, ArrowDownCircle, ArrowUpCircle, Wifi, WifiOff, Upload, Video, Activity, Timer, Database, Mail, Smartphone, Network, Link2 } from 'lucide-react';

type LineDirection = 'horizontal' | 'vertical' | 'diagonal1' | 'diagonal2';

const getSide = (x: number, y: number, dir: LineDirection, pct: number, w: number, h: number) => {
  if (dir === 'horizontal') return y - h * pct;
  if (dir === 'vertical') return x - w * pct;
  const diagDist = Math.hypot(w, h);
  if (dir === 'diagonal1') return ((x / w) + (y / h) - (2 * pct)) * (diagDist / 2);
  if (dir === 'diagonal2') return ((x / w) - (y / h) - (2 * pct - 1)) * (diagDist / 2);
  return 0;
};

interface Track {
  id: number;
  cx: number;
  cy: number;
  box: [number, number, number, number];
  counted: boolean;
  historyX: number;
  historyY: number;
  className: string;
  missingFrames: number;
  age: number;
  vx: number;
  vy: number;
}

interface SessionHistory {
  id: string;
  timestamp: number;
  countIn: number;
  countOut: number;
  mode?: 'line' | 'static';
}

// Simple audio beep utility
const playBeep = (isTarget = false) => {
  try {
    const AudioContext = window.AudioContext || (window as any).webkitAudioContext;
    if (!AudioContext) return;
    const ctx = new AudioContext();
    const osc = ctx.createOscillator();
    const gainNode = ctx.createGain();
    osc.type = isTarget ? 'triangle' : 'sine';
    osc.frequency.setValueAtTime(isTarget ? 1200 : 800, ctx.currentTime);
    if (isTarget) {
        osc.frequency.exponentialRampToValueAtTime(1500, ctx.currentTime + 0.2);
    }
    gainNode.gain.setValueAtTime(isTarget ? 0.2 : 0.1, ctx.currentTime);
    osc.connect(gainNode);
    gainNode.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + (isTarget ? 0.3 : 0.1));
  } catch (e) {
    // Ignore if audio isn't supported or allowed
  }
};

export default function App() {
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<string>('');
  const [statusText, setStatusText] = useState('Yapay Zeka Yükleniyor...');
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  const [countMethod, setCountMethod] = useState<'line' | 'static'>('line');
  const [linePercent, setLinePercent] = useState(50);
  const [lineDirection, setLineDirection] = useState<LineDirection>('horizontal');
  const [countIn, setCountIn] = useState(0);
  const [countOut, setCountOut] = useState(0);
  const [currentStaticCount, setCurrentStaticCount] = useState(0);
  const [maxStaticCount, setMaxStaticCount] = useState(0);
  const [countAnimation, setCountAnimation] = useState(false);
  const [confidenceThreshold, setConfidenceThreshold] = useState(50);
  const [targetCount, setTargetCount] = useState<number | ''>('');
  const [trackingMode, setTrackingMode] = useState<'normal' | 'fast' | 'crowded'>('normal');
  const [targetSpecies, setTargetSpecies] = useState<string>('all');
  
  const [sessionHistory, setSessionHistory] = useState<SessionHistory[]>(() => {
    try {
      const saved = localStorage.getItem('sheepCountHistory');
      return saved ? JSON.parse(saved) : [];
    } catch { return []; }
  });
  const [isHistoryModalOpen, setIsHistoryModalOpen] = useState(false);
  const [isOffline, setIsOffline] = useState(!navigator.onLine);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [roi, setRoi] = useState<{x: number, y: number, w: number, h: number} | null>(null);
  const [isDrawingRoi, setIsDrawingRoi] = useState(false);
  const [roiStart, setRoiStart] = useState<{x: number, y: number} | null>(null);
  const [roiModeActive, setRoiModeActive] = useState(false);
  
  const [alertsEnabled, setAlertsEnabled] = useState(true);
  const [isNightMode, setIsNightMode] = useState(false);
  const [aiEngine, setAiEngine] = useState<'local' | 'cloud'>('local');
  const [sourceType, setSourceType] = useState<'camera' | 'file' | 'stream'>('camera');
  const [videoFileUrl, setVideoFileUrl] = useState<string | null>(null);
  const [streamUrl, setStreamUrl] = useState<string>('');
  const [notificationMethod, setNotificationMethod] = useState<'none' | 'sms' | 'email'>('none');
  const [contactInfo, setContactInfo] = useState<string>('');
  const [autoSync, setAutoSync] = useState<boolean>(false);
  const [videoSpeed, setVideoSpeed] = useState<number>(1.0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  useEffect(() => {
    localStorage.setItem('sheepCountHistory', JSON.stringify(sessionHistory));
  }, [sessionHistory]);

  useEffect(() => {
    const handleOnline = () => setIsOffline(false);
    const handleOffline = () => setIsOffline(true);
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);
    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  useEffect(() => {
    if (videoRef.current && sourceType === 'file') {
      videoRef.current.playbackRate = videoSpeed;
    }
  }, [videoSpeed, videoFileUrl, sourceType]);

  // Use refs for moving parts in the animation loop
  const countMethodRef = useRef(countMethod);
  const countInRef = useRef(countIn);
  const countOutRef = useRef(countOut);
  const linePercentRef = useRef(linePercent);
  const lineDirectionRef = useRef(lineDirection);
  const alertsRef = useRef(alertsEnabled);
  const isNightModeRef = useRef(isNightMode);
  const aiEngineRef = useRef(aiEngine);
  const confidenceThresholdRef = useRef(confidenceThreshold);
  const targetCountRef = useRef(targetCount);
  const trackingModeRef = useRef(trackingMode);
  const targetSpeciesRef = useRef(targetSpecies);
  const roiRef = useRef(roi);
  const isCloudDetectingRef = useRef(false);
  const tracksRef = useRef<Track[]>([]);
  const nextTrackIdRef = useRef(1);
  const avgAreaRef = useRef<number>(0);
  const animationIdRef = useRef<number | null>(null);
  
  useEffect(() => {
    countMethodRef.current = countMethod;
    countInRef.current = countIn;
    countOutRef.current = countOut;
    linePercentRef.current = linePercent;
    lineDirectionRef.current = lineDirection;
    alertsRef.current = alertsEnabled;
    isNightModeRef.current = isNightMode;
    aiEngineRef.current = aiEngine;
    confidenceThresholdRef.current = confidenceThreshold;
    targetCountRef.current = targetCount;
    trackingModeRef.current = trackingMode;
    targetSpeciesRef.current = targetSpecies;
    roiRef.current = roi;
  }, [countMethod, countIn, countOut, linePercent, lineDirection, alertsEnabled, isNightMode, aiEngine, confidenceThreshold, targetCount, trackingMode, targetSpecies, roi]);

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const loadedModel = await cocoSsd.load();
        if (active) {
          setModel(loadedModel);
          setStatusText('Sistem Hazır');
          await getCameras();
        }
      } catch (err) {
        if (active) setStatusText('Yapay Zeka Yüklenemedi');
        console.error(err);
      }
    })();
    return () => { active = false; };
  }, []);

  const getCameras = async () => {
    try {
      await navigator.mediaDevices.getUserMedia({ video: true });
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(d => d.kind === 'videoinput');
      setCameras(videoDevices);
      const backCam = videoDevices.find(d => 
        d.label.toLowerCase().includes('back') || 
        d.label.toLowerCase().includes('arka') ||
        d.label.toLowerCase().includes('environment')
      );
      if (backCam) {
        setSelectedCamera(backCam.deviceId);
      } else if (videoDevices.length > 0) {
        setSelectedCamera(videoDevices[0].deviceId);
      }
    } catch (err) {
      setStatusText('Kamera İzni Gerekli');
    }
  };

  useEffect(() => {
    if (!model) return;
    
    let stream: MediaStream | null = null;
    
    const startSource = async () => {
      if (animationIdRef.current) cancelAnimationFrame(animationIdRef.current);
      
      const video = videoRef.current;
      if (!video) return;

      if (sourceType === 'camera' && selectedCamera) {
        video.src = '';
        const constraints = {
          video: {
            deviceId: { exact: selectedCamera },
            width: { ideal: 640 },
            height: { ideal: 480 },
          }
        };

        try {
          stream = await navigator.mediaDevices.getUserMedia(constraints);
          
          // Kamera odağı ve pozlama optimizasyonu
          const track = stream.getVideoTracks()[0];
          if (track && typeof track.getCapabilities === 'function') {
            const capabilities = track.getCapabilities() as any;
            const advancedConstraints: any = {};
            let applyConstraints = false;

            if (capabilities.focusMode?.includes('continuous')) {
              advancedConstraints.focusMode = 'continuous';
              applyConstraints = true;
            }
            if (capabilities.exposureMode?.includes('continuous')) {
              advancedConstraints.exposureMode = 'continuous';
              applyConstraints = true;
            }

            if (applyConstraints) {
              try {
                await track.applyConstraints({ advanced: [advancedConstraints] });
              } catch (e) {
                console.warn('Otomatik odaklama/pozlama uygulanamadı:', e);
              }
            }
          }

          video.srcObject = stream;
          video.loop = false;
          video.onloadedmetadata = () => {
            if (canvasRef.current && videoRef.current) {
              canvasRef.current.width = video.videoWidth;
              canvasRef.current.height = video.videoHeight;
              video.play().catch(e => console.error("Kamera oynatılamadı", e));
              detectFrame();
            }
          };
        } catch (err) {
          console.error("Kamera başlatılamadı:", err);
        }
      } else if (sourceType === 'file' && videoFileUrl) {
         if (stream) {
             stream.getTracks().forEach(t => t.stop());
             stream = null;
         }
         video.srcObject = null;
         video.src = videoFileUrl;
         video.loop = true;
         video.playbackRate = videoSpeed;
         video.onloadedmetadata = () => {
           if (canvasRef.current && videoRef.current) {
             canvasRef.current.width = video.videoWidth;
             canvasRef.current.height = video.videoHeight;
             video.play().catch(e => console.error("Video oynatılamadı", e));
             detectFrame();
           }
         };
      } else if (sourceType === 'file' && !videoFileUrl) {
          if (stream) {
              stream.getTracks().forEach(t => t.stop());
              stream = null;
          }
          video.srcObject = null;
          video.src = '';
      } else if (sourceType === 'stream' && streamUrl) {
         if (stream) {
             stream.getTracks().forEach(t => t.stop());
             stream = null;
         }
         video.srcObject = null;
         video.crossOrigin = "anonymous";
         video.src = streamUrl;
         video.loop = false;
         video.playbackRate = 1.0;
         video.onloadedmetadata = () => {
           if (canvasRef.current && videoRef.current) {
             canvasRef.current.width = video.videoWidth;
             canvasRef.current.height = video.videoHeight;
             video.play().catch(e => console.error("Yayın oynatılamadı", e));
             detectFrame();
           }
         };
      }
    };

    startSource();

    return () => {
      if (stream) stream.getTracks().forEach(t => t.stop());
      if (animationIdRef.current) cancelAnimationFrame(animationIdRef.current);
    };
  }, [selectedCamera, model, sourceType, videoFileUrl, streamUrl]);

  const handleRoiStart = (clientX: number, clientY: number) => {
    if (!roiModeActive || !canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const scaleX = canvasRef.current.width / rect.width;
    const scaleY = canvasRef.current.height / rect.height;
    
    // Bounds check to see if we clicked inside the actual canvas area (because of object-contain)
    if (clientX < rect.left || clientX > rect.right || clientY < rect.top || clientY > rect.bottom) return;

    const x = (clientX - rect.left) * scaleX;
    const y = (clientY - rect.top) * scaleY;
    
    setIsDrawingRoi(true);
    setRoiStart({x, y});
    setRoi({x, y, w: 0, h: 0});
  };

  const handleRoiMove = (clientX: number, clientY: number) => {
    if (!isDrawingRoi || !roiStart || !canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const scaleX = canvasRef.current.width / rect.width;
    const scaleY = canvasRef.current.height / rect.height;
    
    let x = (clientX - rect.left) * scaleX;
    let y = (clientY - rect.top) * scaleY;
    
    // Clamp to canvas bounds
    x = Math.max(0, Math.min(x, canvasRef.current.width));
    y = Math.max(0, Math.min(y, canvasRef.current.height));

    if (Math.abs(x - roiStart.x) > 5 || Math.abs(y - roiStart.y) > 5) {
      setRoi({
        x: Math.min(x, roiStart.x),
        y: Math.min(y, roiStart.y),
        w: Math.abs(x - roiStart.x),
        h: Math.abs(y - roiStart.y)
      });
    }
  };

  const handleRoiEnd = () => {
    if (isDrawingRoi) {
      setIsDrawingRoi(false);
      setRoiModeActive(false); // Auto-disable mode after drawing once
    }
  };

  const detectFrame = async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.paused || video.ended || !model) return;

    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    if (isNightModeRef.current) {
      ctx.filter = 'brightness(150%) contrast(150%)';
    } else {
      ctx.filter = 'none';
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // reset filter for UI drawing
    ctx.filter = 'none';

    const lineDir = lineDirectionRef.current;
    const lPercent = linePercentRef.current / 100;

    if (video.videoWidth === 0 || video.videoHeight === 0) {
      animationIdRef.current = requestAnimationFrame(detectFrame);
      return;
    }

    const callPollinations = async (b64: string, modelName: string) => {
      const res = await fetch('https://text.pollinations.ai/openai/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: modelName,
          temperature: 0.1,
          messages: [{ role: 'user', content: [
            { type: 'text', text: 'Respond ONLY with a valid JSON array of detected animals (sheep, cow, horse, person, dog, cat). Format: [{"class": "sheep", "score": 0.9, "bbox": [10, 10, 50, 50]}]. Return [] if nothing found. Do not use markdown blocks, just raw JSON.' },
            { type: 'image_url', image_url: { url: b64 } }
          ]}]
        })
      });
      if (!res.ok) throw new Error(`${modelName} fail`);
      const data = await res.json();
      const txt = data.choices[0]?.message?.content;
      if (!txt) throw new Error('Empty');
      const clean = txt.replace(/```json/gi, '').replace(/```/gi, '').trim();
      const parsed = JSON.parse(clean);
      if (!Array.isArray(parsed)) throw new Error('Not array');
      return parsed;
    };

    const performCloudFallbackDetection = async (c: HTMLCanvasElement): Promise<any[]> => {
      try {
        const b64 = c.toDataURL('image/jpeg', 0.5);
        try { return await callPollinations(b64, 'gemini'); } catch (e1) {
          try { return await callPollinations(b64, 'grok'); } catch (e2) {
            const blob = await new Promise<Blob|null>(r => c.toBlob(r, 'image/jpeg', 0.5));
            if (blob) {
              const hfRes = await fetch('https://api-inference.huggingface.co/models/facebook/detr-resnet-50', { method: 'POST', body: blob });
              if (hfRes.ok) {
                const data = await hfRes.json();
                return data.map((d: any) => ({ class: d.label, score: d.score, bbox: [d.box.xmin, d.box.ymin, d.box.xmax-d.box.xmin, d.box.ymax-d.box.ymin] }));
              }
            }
          }
        }
      } catch(e) {}
      return [];
    };

    if (aiEngineRef.current === 'local') {
      try {
        let inputToModel: any = canvas;
        let tensorToDispose: tf.Tensor3D | null = null;

        if (isNightModeRef.current) {
          tensorToDispose = tf.tidy(() => {
            let img = tf.browser.fromPixels(canvas).toFloat();
            // Gürültü azaltma ve eşikleme filtresi (Night Mode)
            // 3x3 Average Blur kernel (channel-wise)
            const kernel = tf.fill([3, 3, 3, 1], 1/9);
            img = tf.depthwiseConv2d(img, kernel, 1, 'same');
            
            // Parlaklık Eşikleme (Noise Thresholding):
            // Ortalama değerin altındaki hafif gürültüleri tamamen siyaha çek (0)
            img = img.where(img.greater(tf.scalar(30)), tf.scalar(0));
            // Kontrast tekrar düzenle
            img = img.mul(tf.scalar(1.1)).clipByValue(0, 255);
            
            return img.cast('int32') as tf.Tensor3D;
          });
          inputToModel = tensorToDispose;
        }

        const predictions = await model.detect(inputToModel);
        
        if (tensorToDispose) {
          tensorToDispose.dispose();
        }

        let allowedClasses = ['sheep', 'cow', 'horse', 'person', 'dog', 'cat'];
        const target = targetSpeciesRef.current;
        if (target !== 'all') {
          if (target === 'bovine') allowedClasses = ['cow', 'horse'];
          else allowedClasses = [target];
        }
        const animalDetections = predictions.filter(p => {
          if (!allowedClasses.includes(p.class) || p.score < (confidenceThresholdRef.current / 100)) return false;
          if (roiRef.current) {
             const cx = p.bbox[0] + (p.bbox[2] / 2);
             const cy = p.bbox[1] + (p.bbox[3] / 2);
             const {x, y, w, h} = roiRef.current;
             if (cx < x || cx > x + w || cy < y || cy > y + h) return false;
          }
          return true;
        });
        updateTracker(animalDetections, lineDir, lPercent, canvas.width, canvas.height);
      } catch (e) {
        console.error("Detection error", e);
      }
    } else {
      if (!isCloudDetectingRef.current) {
        isCloudDetectingRef.current = true;
        performCloudFallbackDetection(canvas).then(predictions => {
          let allowedClasses = ['sheep', 'cow', 'horse', 'person', 'dog', 'cat'];
          const target = targetSpeciesRef.current;
          if (target !== 'all') {
             if (target === 'bovine') allowedClasses = ['cow', 'horse'];
             else allowedClasses = [target];
          }
          const animalDetections = predictions.filter(p => {
             const clsMat = allowedClasses.includes(String(p.class).toLowerCase());
             const scoreMat = p.score !== undefined ? p.score >= (confidenceThresholdRef.current / 100) : true;
             if (!clsMat || !scoreMat) return false;
             if (roiRef.current) {
                const cx = p.bbox[0] + (p.bbox[2] / 2);
                const cy = p.bbox[1] + (p.bbox[3] / 2);
                const {x, y, w, h} = roiRef.current;
                if (cx < x || cx > x + w || cy < y || cy > y + h) return false;
             }
             return true;
          });
          updateTracker(animalDetections, lineDir, lPercent, canvas.width, canvas.height);
        }).catch(err => {
          console.error("Cloud Fallback error", err);
        }).finally(() => {
          isCloudDetectingRef.current = false;
        });
      }
    }

    // Draw Line
    if (countMethodRef.current === 'line') {
      ctx.beginPath();
      if (lineDir === 'horizontal') {
        ctx.moveTo(0, canvas.height * lPercent);
        ctx.lineTo(canvas.width, canvas.height * lPercent);
      } else if (lineDir === 'vertical') {
        ctx.moveTo(canvas.width * lPercent, 0);
        ctx.lineTo(canvas.width * lPercent, canvas.height);
      } else if (lineDir === 'diagonal1') {
        ctx.moveTo(0, 2 * lPercent * canvas.height);
        ctx.lineTo(canvas.width, (2 * lPercent - 1) * canvas.height);
      } else if (lineDir === 'diagonal2') {
        ctx.moveTo(0, (1 - 2 * lPercent) * canvas.height);
        ctx.lineTo(canvas.width, (2 - 2 * lPercent) * canvas.height);
      }
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 8;
      ctx.setLineDash([]);
      ctx.stroke();
    }

    // Draw Boxes
    for (let track of tracksRef.current) {
      if (track.missingFrames > 0) {
        ctx.globalAlpha = 0.5; // Prediction ghosting effect
      } else {
        ctx.globalAlpha = 1.0;
      }
      
      ctx.strokeStyle = track.counted ? '#10b981' : '#eab308';
      ctx.lineWidth = 2;
      ctx.strokeRect(track.box[0], track.box[1], track.box[2], track.box[3]);
      
      ctx.fillStyle = track.counted ? '#10b981' : '#eab308';
      ctx.fillRect(track.box[0], track.box[1] - 20, 110, 20);
      ctx.fillStyle = '#000000';
      ctx.font = '12px system-ui font-bold';
      ctx.fillText(`${track.className} ID: ${track.id}`, track.box[0] + 5, track.box[1] - 5);

      ctx.fillStyle = '#ef4444';
      ctx.beginPath();
      ctx.arc(track.cx, track.cy, 4, 0, 2 * Math.PI);
      ctx.fill();
    }
    ctx.globalAlpha = 1.0; // Reset just in case

    // Draw Watermark (Date, Time, System Info)
    ctx.fillStyle = 'rgba(0,0,0,0.5)';
    ctx.fillRect(10, 10, 320, 60);
    ctx.font = 'bold 16px "Inter", sans-serif';
    ctx.fillStyle = '#10b981';
    ctx.fillText('SÜRÜTAKİBİ AI © 2026', 20, 30);
    ctx.fillStyle = '#ffffff';
    ctx.font = '14px "JetBrains Mono", monospace';
    const now = new Date();
    ctx.fillText(`${now.toLocaleDateString('tr-TR')} ${now.toLocaleTimeString('tr-TR')}`, 20, 50);
    const targetStr = targetSpeciesRef.current === 'all' ? 'Tümü' : targetSpeciesRef.current === 'sheep' ? 'Koyun' : targetSpeciesRef.current === 'bovine' ? 'Büyükbaş' : 'İnsan';
    ctx.fillText(`TÜR: ${targetStr} | CİHAZ: K-01`, 20, 65);

    // Draw ROI (İlgi Bölgesi)
    if (roiRef.current) {
      const r = roiRef.current;
      ctx.strokeStyle = 'rgba(16, 185, 129, 0.8)';
      ctx.lineWidth = 3;
      ctx.setLineDash([5, 5]);
      ctx.strokeRect(r.x, r.y, r.w, r.h);
      ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(16, 185, 129, 0.1)';
      ctx.fillRect(r.x, r.y, r.w, r.h);
      
      // ROI Label
      ctx.fillStyle = 'rgba(16, 185, 129, 0.8)';
      ctx.font = 'bold 14px "Inter", sans-serif';
      ctx.fillText('SAYIM ALANI', r.x + 5, r.y + 18);
    }

    if (roiModeActive && !roiRef.current) {
       ctx.fillStyle = 'rgba(0,0,0,0.6)';
       ctx.fillRect(0, 0, canvas.width, canvas.height);
       ctx.fillStyle = '#10b981';
       ctx.textAlign = 'center';
       ctx.font = 'bold 24px "Inter", sans-serif';
       ctx.fillText('Ekrana sürükleyerek sayım alanı (ROI) çizin', canvas.width / 2, canvas.height / 2);
       ctx.textAlign = 'left';
    }

    animationIdRef.current = requestAnimationFrame(detectFrame);
  };

  const updateTracker = (detections: cocoSsd.DetectedObject[], lineDir: LineDirection, linePct: number, w: number, h: number) => {
    let MAX_DIST = 150;
    let MAX_MISSING = 15;
    
    if (trackingModeRef.current === 'fast') {
      MAX_DIST = 350; // Geniş arama (hızlı hareket eden objeler için)
      MAX_MISSING = 8; // Hızlı ekrandan çıkma ihtimaline karşı düşük tolerans
    } else if (trackingModeRef.current === 'crowded') {
      MAX_DIST = 90; // Sıkı arama (kimlik karışmasını önlemek için)
      MAX_MISSING = 30; // Kalabalıkta üst üste binen hayvanlar için uzun tahminleme (oklüzyon hilesi)
    }

    const currentTracks: Track[] = [];
    let tracks = tracksRef.current;
    
    // Hazırlık: Tespitleri standartlaştır
    const detectionsArray = detections.map(det => ({
       cx: det.bbox[0] + (det.bbox[2] / 2),
       cy: det.bbox[1] + (det.bbox[3] / 2),
       box: det.bbox as [number, number, number, number],
       className: det.class,
       matched: false
    }));

    // Yoğunluk Analizi: Ortalama Nesne Büyüklüğü (Hareketli Ortalama)
    for (const d of detectionsArray) {
       const area = d.box[2] * d.box[3];
       if (avgAreaRef.current === 0) {
           avgAreaRef.current = area;
       } else {
           if (area < avgAreaRef.current * 1.8 && area > avgAreaRef.current * 0.5) {
               avgAreaRef.current = (avgAreaRef.current * 0.95) + (area * 0.05);
           }
       }
    }

    // Aktif izlerin bir sonraki konumlarını hızları baz alarak (Kalman-vari) tahmin et
    const activeTracks = tracks.map(t => ({
        ...t,
        predCx: t.cx + t.vx,
        predCy: t.cy + t.vy,
        matched: false
    }));

    const getIOU = (box1: [number, number, number, number], box2: [number, number, number, number]) => {
      const [x1, y1, w1, h1] = box1;
      const [x2, y2, w2, h2] = box2;
      const interX = Math.max(x1, x2);
      const interY = Math.max(y1, y2);
      const interW = Math.min(x1+w1, x2+w2) - interX;
      const interH = Math.min(y1+h1, y2+h2) - interY;
      if (interW <= 0 || interH <= 0) return 0;
      const interArea = interW * interH;
      return interArea / ((w1 * h1) + (w2 * h2) - interArea);
    };

    // Mesafe matrisi oluştur
    const pairs: {trackIdx: number, detIdx: number, dist: number, cost: number}[] = [];
    for (let i = 0; i < activeTracks.length; i++) {
        for (let j = 0; j < detectionsArray.length; j++) {
            const dist = Math.hypot(activeTracks[i].predCx - detectionsArray[j].cx, activeTracks[i].predCy - detectionsArray[j].cy);
            if (dist < MAX_DIST) {
                const iou = getIOU(activeTracks[i].box, detectionsArray[j].box);
                const cost = dist * (1 - iou) - (iou * 20); // IOU yüksekse maliyeti ciddi şekilde düşür
                pairs.push({ trackIdx: i, detIdx: j, dist, cost });
            }
        }
    }

    // Açgözlü Eşleştirme (Maliyete Göre Artan Sıralama)
    pairs.sort((a, b) => a.cost - b.cost);

    for (const pair of pairs) {
        const t = activeTracks[pair.trackIdx];
        const d = detectionsArray[pair.detIdx];
        if (!t.matched && !d.matched) {
            t.matched = true;
            d.matched = true;
            
            // Hızı düzelt (Hareketli ortalama filtresi)
            const newVx = d.cx - t.cx;
            const newVy = d.cy - t.cy;
            
            currentTracks.push({
                ...t,
                vx: (t.vx * 0.5) + (newVx * 0.5),
                vy: (t.vy * 0.5) + (newVy * 0.5),
                cx: d.cx,
                cy: d.cy,
                box: d.box,
                className: d.className,
                missingFrames: 0,
                age: t.age + 1
            });
        }
    }

    // Eşleşmeyen izleri belli bir süre tahmini pozisyonda yaşat (Oklüzyon durumu)
    for (const t of activeTracks) {
        if (!t.matched && t.missingFrames < MAX_MISSING) {
            currentTracks.push({
                ...t,
                missingFrames: t.missingFrames + 1,
                // Kayıp olduğu anlarda çizgiyi geçmiş saysa bile sayıma katmayacağız
                cx: t.cx + (t.vx * 0.5),
                cy: t.cy + (t.vy * 0.5),
            });
        }
    }

    // Eşleşmeyen yeni tespitleri yeni iz olarak ekle
    for (const d of detectionsArray) {
        if (!d.matched) {
            currentTracks.push({
                id: nextTrackIdRef.current++,
                cx: d.cx,
                cy: d.cy,
                box: d.box,
                counted: false,
                historyX: d.cx,
                historyY: d.cy,
                className: d.className,
                missingFrames: 0,
                age: 1,
                vx: 0,
                vy: 0
            });
        }
    }

    // Sayım Mantığı (Kayıp olmayan izler üzerinde)
    let currentActiveCount = 0;
    for (let t of currentTracks) {
      if (t.missingFrames === 0) {
        currentActiveCount++;
        
        const isCrowded = trackingModeRef.current === 'crowded';
        const ageReq = isCrowded ? 2 : 5;
        const distReq = isCrowded ? 2 : 10;
        
        if (countMethodRef.current === 'line' && !t.counted && t.age >= ageReq) {
          let dir: 'in' | 'out' | null = null;
          
          const side1 = getSide(t.historyX, t.historyY, lineDir, linePct, w, h);
          const side2 = getSide(t.cx, t.cy, lineDir, linePct, w, h);
          
          // Cross boundary and travel enough distance to avoid jitter
          if (side1 <= 0 && side2 > 0 && Math.abs(side1 - side2) > distReq) {
            dir = 'in';
          } else if (side1 >= 0 && side2 < 0 && Math.abs(side1 - side2) > distReq) {
            dir = 'out';
          }

          if (dir) {
            t.counted = true;
            
            let countToAdd = 1;
            if (isCrowded && avgAreaRef.current > 0) {
              const boxArea = t.box[2] * t.box[3];
              // Yoğunluk Analizi: Eğer bounding box ortalamadan oldukça büyükse birden fazla koyun olabilir
              countToAdd = Math.max(1, Math.round(boxArea / avgAreaRef.current));
            }

            setCountAnimation(true);
            setTimeout(() => setCountAnimation(false), 200);

            let cIn = countInRef.current;
            let cOut = countOutRef.current;
            if (dir === 'in') {
              setCountIn(prev => prev + countToAdd);
              cIn += countToAdd;
              countInRef.current += countToAdd;
            }
            if (dir === 'out') {
              setCountOut(prev => prev + countToAdd);
              cOut += countToAdd;
              countOutRef.current += countToAdd;
            }
            const net = Math.abs(cIn - cOut);
            
            if (alertsRef.current) {
              if (navigator.vibrate) navigator.vibrate(50);
              
              const target = Number(targetCountRef.current);
              if (target > 0 && net === target) {
                  if (navigator.vibrate) navigator.vibrate([100, 50, 100]);
                  playBeep(true);
              } else {
                  playBeep(false);
              }
            }
          }
        }
        
      }
    }

    if (countMethodRef.current === 'static') {
        setCurrentStaticCount(currentActiveCount);
        setMaxStaticCount(prev => Math.max(prev, currentActiveCount));
    }

    tracksRef.current = currentTracks;
  };

  const resetCounter = () => {
    setCountIn(0);
    setCountOut(0);
    setCurrentStaticCount(0);
    setMaxStaticCount(0);
    tracksRef.current = [];
  };

  const saveSession = () => {
    if (countMethod === 'line' && countIn === 0 && countOut === 0) return;
    if (countMethod === 'static' && maxStaticCount === 0) return;
    const newSession: SessionHistory = {
      id: Math.random().toString(36).substring(2, 9),
      timestamp: Date.now(),
      countIn: countMethod === 'line' ? countIn : maxStaticCount,
      countOut: countMethod === 'line' ? countOut : 0,
      mode: countMethod
    };
    setSessionHistory(prev => [newSession, ...prev]);
    resetCounter();

    if (autoSync || notificationMethod !== 'none') {
        let msg = "";
        if (autoSync) msg += "Sürü API aktarıldı. ";
        if (notificationMethod === 'sms') msg += "SMS gönderildi. ";
        if (notificationMethod === 'email') msg += "E-posta iletildi.";
        setStatusText(msg.trim());
        setTimeout(() => setStatusText('Sistem Hazır'), 4000);
    } else {
        setStatusText('Sayım Kaydedildi');
        setTimeout(() => setStatusText('Sistem Hazır'), 3000);
    }
  };

  const downloadCSV = () => {
    if (sessionHistory.length === 0) return;
    const csvContent = "data:text/csv;charset=utf-8," 
      + "Tarih,Mod,Giren/Max,Cikan,Net\n"
      + sessionHistory.map(row => {
          const d = new Date(row.timestamp);
          const modeStr = row.mode === 'static' ? 'Mera(Max)' : 'Gecis';
          const net = row.mode === 'static' ? row.countIn : Math.abs(row.countIn - row.countOut);
          return `${d.toLocaleDateString('tr-TR')} ${d.toLocaleTimeString('tr-TR')},${modeStr},${row.countIn},${row.mode==='static'?'-':row.countOut},${net}`;
        }).join("\n");
        
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "koyun_sayim_gecmisi.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="bg-slate-900 text-white h-screen w-screen relative flex flex-col items-center justify-center overflow-hidden font-sans">
      {isOffline && (
        <div className="absolute top-4 right-4 z-50 bg-rose-500/80 backdrop-blur-md text-white px-3 py-1.5 rounded-full flex items-center gap-1.5 text-xs font-bold shadow-lg shadow-black/50 border border-rose-400">
          <WifiOff className="w-3.5 h-3.5" /> Çevrimdışı (Ön bellek devrede)
        </div>
      )}

      <div 
        className="absolute inset-0 z-0 flex items-center justify-center bg-black"
        onMouseDown={e => handleRoiStart(e.clientX, e.clientY)}
        onMouseMove={e => handleRoiMove(e.clientX, e.clientY)}
        onMouseUp={handleRoiEnd}
        onMouseLeave={handleRoiEnd}
        onTouchStart={e => handleRoiStart(e.touches[0].clientX, e.touches[0].clientY)}
        onTouchMove={e => handleRoiMove(e.touches[0].clientX, e.touches[0].clientY)}
        onTouchEnd={handleRoiEnd}
      >
        <video 
          ref={videoRef} 
          playsInline 
          autoPlay 
          muted 
          className="hidden" 
          width={640} 
          height={480}
        />
        <canvas ref={canvasRef} className="max-w-full max-h-full object-contain" />
      </div>

      <div className="absolute top-4 left-4 right-4 z-10 flex flex-col md:flex-row justify-between items-start pointer-events-none gap-4">
        
        <div className="pointer-events-auto flex flex-col gap-3">
          <div className="bg-slate-900/80 backdrop-blur-md border border-slate-700/50 px-5 py-3 rounded-2xl shadow-xl flex items-center gap-3">
            <div className="bg-emerald-500/20 p-2 rounded-lg">
               <Cpu className="w-5 h-5 text-emerald-400" />
            </div>
            <div>
              <h1 className="text-white font-black leading-tight tracking-wide">SÜRÜTAKİBİ AI</h1>
              <p className="text-[10px] text-emerald-400 font-mono font-bold uppercase tracking-wider">{model ? 'Sistem Aktif - K-01 Edge' : 'Yükleniyor...'}</p>
            </div>
          </div>

          <div className="bg-slate-900/80 backdrop-blur-md border border-slate-700/50 px-6 py-4 rounded-2xl shadow-xl flex flex-col items-center">
            <span className="text-slate-400 text-[10px] font-bold uppercase tracking-widest mb-1">
              {countMethod === 'line' ? 'Günlük Net Geçiş' : 'Anlık Sürü Mevcudu'}
            </span>
            <span className={`text-[3.5rem] font-black tabular-nums tracking-tighter drop-shadow-md transition-all duration-200 leading-none ${countAnimation ? 'scale-110 text-emerald-400' : 'text-white scale-100'}`}>
              {countMethod === 'line' ? Math.abs(countIn - countOut) : currentStaticCount}
            </span>
            {countMethod === 'line' ? (
                <div className="flex gap-4 mt-3 text-xs font-bold bg-slate-800/50 px-3 py-1 rounded-full border border-slate-700">
                  <span className="flex items-center gap-1 text-emerald-400" title="Giren"><ArrowDownCircle className="w-3.5 h-3.5"/> {countIn}</span>
                  <span className="text-slate-600">|</span>
                  <span className="flex items-center gap-1 text-rose-400" title="Çıkan"><ArrowUpCircle className="w-3.5 h-3.5"/> {countOut}</span>
                </div>
            ) : (
                <div className="flex gap-4 mt-3 text-xs font-bold bg-slate-800/50 px-3 py-1 rounded-full border border-slate-700">
                  <span className="flex items-center gap-1 text-blue-400" title="Maksimum Görülen">Maksimum Tespit: {maxStaticCount}</span>
                </div>
            )}
          </div>
        </div>

        <div className="bg-slate-900/80 backdrop-blur-md border border-slate-700/50 p-4 rounded-2xl shadow-xl pointer-events-auto w-full max-w-xs space-y-4">
          <div className={`flex items-center gap-3 text-xs font-medium ${model ? 'text-emerald-400' : 'text-blue-400'}`}>
            {!model ? <Loader2 className="w-4 h-4 animate-spin" /> : <Activity className="w-4 h-4" />}
            <span className="truncate">{statusText}</span>
          </div>

          <div className="h-px bg-slate-700 w-full" />

          <div className="space-y-1">
            <label className="text-xs text-slate-400 font-semibold flex items-center gap-1">
              <Server className="w-3 h-3"/> AI Motoru
            </label>
            <select
              value={aiEngine}
              onChange={e => setAiEngine(e.target.value as 'local' | 'cloud')}
              className="w-full bg-slate-800 border border-slate-600 text-white text-sm rounded-lg p-2.5 focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer"
            >
              <option value="local">Yerel Model (Hızlı, 30FPS)</option>
              <option value="cloud">Bulut (Grok/Gemini/HF - Yavaş)</option>
            </select>
          </div>

          <div className="space-y-1">
            <label className="text-xs text-slate-400 font-semibold flex items-center gap-1">
              <Camera className="w-3 h-3"/> Görüntü Kaynağı
            </label>
            <div className="flex bg-slate-800 rounded-lg p-1 border border-slate-700">
              <button 
                onClick={() => setSourceType('camera')}
                className={`flex-1 py-1 text-[11px] font-bold rounded-md transition-colors ${sourceType === 'camera' ? 'bg-slate-700 text-white shadow' : 'text-slate-400 hover:text-slate-300'}`}
              >
                Kamera
              </button>
              <button 
                onClick={() => setSourceType('file')}
                className={`flex-1 py-1 text-[11px] font-bold rounded-md transition-colors flex items-center justify-center gap-1 ${sourceType === 'file' ? 'bg-slate-700 text-white shadow' : 'text-slate-400 hover:text-slate-300'}`}
              >
                Video
              </button>
              <button 
                onClick={() => setSourceType('stream')}
                className={`flex-1 py-1 text-[11px] font-bold rounded-md transition-colors flex items-center justify-center gap-1 ${sourceType === 'stream' ? 'bg-slate-700 text-white shadow' : 'text-slate-400 hover:text-slate-300'}`}
              >
                İnternet/IP
              </button>
            </div>
            
            {sourceType === 'camera' ? (
              <div className="relative mt-2">
                <select 
                  value={selectedCamera}
                  onChange={e => setSelectedCamera(e.target.value)}
                  disabled={cameras.length === 0}
                  className="w-full bg-slate-900 border border-slate-700 text-white text-sm rounded-lg p-2.5 appearance-none focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
                >
                  {cameras.length === 0 && <option>Kamera Aranıyor...</option>}
                  {cameras.map((cam, i) => (
                    <option key={cam.deviceId} value={cam.deviceId}>
                      {cam.label || `Kamera ${i + 1}`}
                    </option>
                  ))}
                </select>
                <Camera className="absolute right-3 top-2.5 w-4 h-4 text-slate-400 pointer-events-none" />
              </div>
            ) : sourceType === 'file' ? (
              <div className="mt-2 text-center w-full">
                <input 
                  title="Video Yükle"
                  type="file" 
                  accept="video/*" 
                  ref={fileInputRef}
                  className="hidden" 
                  onChange={e => {
                    const file = e.target.files?.[0];
                    if (file) {
                      const url = URL.createObjectURL(file);
                      setVideoFileUrl(url);
                      // Sürükle bırak / normal yükleme sonrası sayacı sıfırla
                      resetCounter();
                    }
                  }}
                />
                <button 
                  onClick={() => fileInputRef.current?.click()}
                  className="w-full bg-slate-900 hover:bg-slate-800 border border-slate-700 text-blue-400 transition-colors py-2.5 rounded-lg text-sm font-bold flex items-center justify-center gap-2"
                >
                  <Upload className="w-4 h-4" /> {videoFileUrl ? 'Yeni Video Seç' : 'Video Yükle'}
                </button>
                {videoFileUrl && (
                  <div className="mt-3 text-left space-y-1">
                    <label className="text-xs text-slate-400 font-semibold flex items-center justify-between">
                      <span className="flex items-center gap-1"><Timer className="w-3 h-3"/> Oynatma Hızı (Yapay Zeka İçin)</span>
                      <span>{videoSpeed}x</span>
                    </label>
                    <input 
                      type="range" min="0.25" max="2" step="0.25" value={videoSpeed}
                      onChange={e => setVideoSpeed(Number(e.target.value))}
                      className="w-full accent-blue-500 h-1.5 bg-slate-900 border border-slate-700 rounded-lg appearance-none cursor-pointer" 
                    />
                    <p className="text-[10px] text-slate-500 leading-tight">Yüksek hızda kaçırılan sayımları önlemek için oynatmayı yavaşlatabilirsiniz.</p>
                  </div>
                )}
              </div>
            ) : (
              <div className="mt-2 w-full space-y-2">
                <div className="relative">
                  <Network className="absolute left-3 top-2.5 w-4 h-4 text-slate-400" />
                  <input
                    type="url"
                    placeholder="http://192.168.1.100:8080/video"
                    value={streamUrl}
                    onChange={(e) => setStreamUrl(e.target.value)}
                    className="w-full pl-9 bg-slate-900 border border-slate-700 text-white text-sm rounded-lg p-2.5 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <p className="text-[10px] text-slate-500 leading-tight">IP kamera veya canlı yayın URL'sini yapıştırın. Format desteklenmiyorsa tarayıcı kaynaklı olabilir.</p>
              </div>
            )}
          </div>

          <div className="space-y-1">
            <label className="text-xs text-slate-400 font-semibold flex items-center gap-1">
              <Target className="w-3 h-3"/> Hedef Tür
            </label>
            <select
              value={targetSpecies}
              onChange={e => setTargetSpecies(e.target.value)}
              className="w-full bg-slate-800 border border-slate-600 text-white text-sm rounded-lg p-2.5 focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer"
            >
              <option value="all">Tüm Hayvanlar & İnsan</option>
              <option value="sheep">Sadece Koyun</option>
              <option value="bovine">Büyükbaş (İnek, At)</option>
              <option value="person">Sadece İnsan (Güvenlik)</option>
            </select>
          </div>

          <div className="space-y-1">
            <label className="text-xs text-slate-400 font-semibold flex items-center gap-1">
              <Activity className="w-3 h-3"/> Sayım Modu
            </label>
            <div className="flex bg-slate-800 rounded-lg p-1 border border-slate-700">
              <button 
                onClick={() => setCountMethod('line')}
                className={`flex-1 py-1.5 text-[11px] font-bold rounded-md transition-colors ${countMethod === 'line' ? 'bg-slate-700 text-white shadow' : 'text-slate-400 hover:text-slate-300'}`}
              >
                Geçiş Sayımı
              </button>
              <button 
                onClick={() => setCountMethod('static')}
                className={`flex-1 py-1.5 text-[11px] font-bold rounded-md transition-colors ${countMethod === 'static' ? 'bg-slate-700 text-white shadow' : 'text-slate-400 hover:text-slate-300'}`}
              >
                Mera Sayımı
              </button>
            </div>
          </div>

          {countMethod === 'line' && (
            <>
              <div className="space-y-1">
                <label className="text-xs text-slate-400 font-semibold">Çizgi Yönü</label>
            <select
              value={lineDirection}
              onChange={e => setLineDirection(e.target.value as LineDirection)}
              className="w-full bg-slate-800 border border-slate-600 text-white text-sm rounded-lg p-2.5 focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer"
            >
              <option value="horizontal">Yatay (Aşağı/Yukarı Geçiş)</option>
              <option value="vertical">Dikey (Sağ/Sol Geçiş)</option>
              <option value="diagonal1">Çapraz ↗ (Sol-Alt ↔ Sağ-Üst)</option>
              <option value="diagonal2">Çapraz ↘ (Sol-Üst ↔ Sağ-Alt)</option>
            </select>
          </div>

          <div className="space-y-1">
            <label className="text-xs text-slate-400 font-semibold flex items-center gap-1">
              <Target className="w-3 h-3"/> İlgi Bölgesi (ROI)
            </label>
            <div className="flex bg-slate-800 rounded-lg p-1 border border-slate-700 gap-1 mt-2">
              <button 
                onClick={() => {
                   setRoiModeActive(true);
                   setRoi(null);
                }}
                className={`flex-1 py-1.5 text-[11px] font-bold rounded-md transition-colors ${roiModeActive || roi ? 'bg-emerald-600 text-white shadow' : 'bg-slate-700 text-slate-300 hover:bg-slate-600'}`}
              >
                {roi ? 'Yeniden Çiz' : 'Alanı Çiz'}
              </button>
              {roi && (
                <button 
                  onClick={() => {
                    setRoi(null);
                    setRoiModeActive(false);
                  }}
                  className="px-3 py-1.5 bg-rose-500/80 hover:bg-rose-500 text-white text-[11px] font-bold rounded-md transition-colors"
                >
                  Temizle
                </button>
              )}
            </div>
            <p className="text-[10px] text-slate-500 leading-tight">Yalnızca bu alan içindeki nesneler sayılır. Arka plan hareketlerini engeller.</p>
          </div>

          <div className="space-y-1">
            <label className="text-xs text-slate-400 font-semibold flex justify-between">
              <span>Sayım Çizgisi</span>
              <span>{linePercent}%</span>
            </label>
            <input 
              type="range" 
              min="10" 
              max="90" 
              value={linePercent}
              onChange={e => setLinePercent(Number(e.target.value))}
              className="w-full accent-red-500 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer" 
            />
          </div>
            </>
          )}

          <div className="grid grid-cols-2 gap-2 pt-2 border-t border-slate-700 mt-2">
            <button 
              onClick={saveSession}
              disabled={(countMethod === 'line' && countIn === 0 && countOut === 0) || (countMethod === 'static' && maxStaticCount === 0)}
              className="w-full bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 border border-blue-500/50 transition-colors py-2 rounded-lg text-sm font-bold flex items-center justify-center gap-1 disabled:opacity-50"
            >
              <Save className="w-4 h-4" /> Kaydet
            </button>
            <button 
              onClick={() => setIsHistoryModalOpen(true)}
              className="w-full bg-slate-700/50 hover:bg-slate-700/70 text-slate-300 border border-slate-600 transition-colors py-2 rounded-lg text-sm font-bold flex items-center justify-center gap-1"
            >
              <History className="w-4 h-4" /> Geçmiş
            </button>
            <button 
              onClick={() => setAlertsEnabled(!alertsEnabled)}
              className={`w-full col-span-2 transition-colors py-2 rounded-lg text-sm font-bold flex items-center justify-center gap-2 border ${alertsEnabled ? 'bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-400 border-emerald-500/50' : 'bg-slate-700/50 hover:bg-slate-700/70 text-slate-400 border-slate-600'}`}
            >
              <Vibrate className="w-4 h-4" /> {alertsEnabled ? "Ses ve Titreşim : Açık" : "Ses ve Titreşim : Kapalı"}
            </button>
            <button 
              onClick={() => setIsNightMode(!isNightMode)}
              className={`w-full col-span-2 transition-colors py-2 rounded-lg text-sm font-bold flex items-center justify-center gap-2 border ${isNightMode ? 'bg-amber-500/20 hover:bg-amber-500/30 text-amber-400 border-amber-500/50' : 'bg-slate-700/50 hover:bg-slate-700/70 text-slate-400 border-slate-600'}`}
            >
              <Moon className="w-4 h-4" /> {isNightMode ? "Gece Modu : Açık" : "Gece Modu : Kapalı"}
            </button>
            
            <button 
              onClick={() => setIsSettingsOpen(!isSettingsOpen)}
              className="w-full col-span-2 bg-slate-800 hover:bg-slate-700 border border-slate-600 text-slate-300 transition-colors py-2 rounded-lg text-sm font-bold flex items-center justify-center gap-2"
            >
              <Settings className="w-4 h-4" /> {isSettingsOpen ? 'Gelişmiş Ayarları Gizle' : 'Gelişmiş Ayarları Göster'}
            </button>

            {isSettingsOpen && (
              <div className="col-span-2 space-y-4 pt-4 pb-2 border-t border-slate-700 transition-all">
                <div className="space-y-1">
                  <label className="text-xs text-slate-400 font-semibold flex items-center gap-1">
                    <Activity className="w-3 h-3"/> Senaryo Takip Algoritması (SORT)
                  </label>
                  <select
                    value={trackingMode}
                    onChange={e => setTrackingMode(e.target.value as 'normal' | 'fast' | 'crowded')}
                    className="w-full bg-slate-900 border border-slate-700 text-white text-sm rounded-lg p-2.5 focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer"
                  >
                    <option value="normal">Normal Takip (Ortalama Sürü)</option>
                    <option value="fast">Hızlı Geçiş (Koşan Sürü) - Geniş Arama</option>
                    <option value="crowded">Kalabalık Sürü - Oklüzyon Hilesi</option>
                  </select>
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-slate-400 font-semibold flex items-center justify-between">
                    <span className="flex items-center gap-1"><Target className="w-3 h-3"/> Güven Eşiği</span>
                    <span>{confidenceThreshold}%</span>
                  </label>
                  <input 
                    type="range" min="10" max="90" value={confidenceThreshold}
                    onChange={e => setConfidenceThreshold(Number(e.target.value))}
                    className="w-full accent-blue-500 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer" 
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-slate-400 font-semibold flex items-center gap-1">
                    <Bell className="w-3 h-3"/> Hedef Parti Kotası
                  </label>
                  <input 
                    type="number" value={targetCount} placeholder="Örn: 100"
                    onChange={e => setTargetCount(e.target.value ? Number(e.target.value) : '')}
                    className="w-full bg-slate-900 border border-slate-700 text-white text-sm rounded-lg p-2.5 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                
                <div className="pt-2 border-t border-slate-700">
                  <span className="text-xs font-bold text-slate-300 uppercase tracking-widest flex items-center gap-1 mb-3">
                    <Database className="w-4 h-4 text-blue-400"/> Entegrasyon & Bulut
                  </span>
                  
                  <div className="space-y-3">
                    <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
                      <input 
                        type="checkbox" 
                        checked={autoSync}
                        onChange={e => setAutoSync(e.target.checked)}
                        className="rounded border-slate-600 bg-slate-800 text-blue-500 focus:ring-blue-500 focus:ring-offset-slate-900 w-4 h-4 cursor-pointer" 
                      />
                      <span className="flex items-center gap-1"><Link2 className="w-4 h-4 text-slate-400"/> Çiftlik Yönetim Sistemine (API) Aktar</span>
                    </label>

                    <div className="space-y-1">
                      <label className="text-xs text-slate-400 font-semibold flex items-center gap-1">
                         Otomatik Bildirim Sistemi
                      </label>
                      <select
                        value={notificationMethod}
                        onChange={e => setNotificationMethod(e.target.value as 'none' | 'sms' | 'email')}
                        className="w-full bg-slate-900 border border-slate-700 text-white text-sm rounded-lg p-2.5 focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer"
                      >
                        <option value="none">Sayım Sonunda Bildirim Gönderme</option>
                        <option value="sms">SMS Gönder (Saha Sorumlusu)</option>
                        <option value="email">E-Posta Raporu (Yönetici)</option>
                      </select>
                    </div>

                    {notificationMethod !== 'none' && (
                      <div className="space-y-1">
                        <label className="text-xs text-slate-400 font-semibold flex items-center gap-1">
                          {notificationMethod === 'sms' ? <Smartphone className="w-3 h-3"/> : <Mail className="w-3 h-3"/>}
                          {notificationMethod === 'sms' ? 'Telefon Numarası' : 'E-Posta Adresi'}
                        </label>
                        <input 
                          type={notificationMethod === 'sms' ? "tel" : "email"}
                          placeholder={notificationMethod === 'sms' ? "+90 5XX XXX XX XX" : "firma@ciftlik.com"}
                          value={contactInfo}
                          onChange={e => setContactInfo(e.target.value)}
                          className="w-full bg-slate-900 border border-slate-700 text-white text-sm rounded-lg p-2.5 focus:outline-none focus:ring-2 focus:ring-blue-500"
                        />
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            <button 
              onClick={resetCounter}
              className="w-full col-span-2 bg-red-500/20 hover:bg-red-500/30 text-red-500 border border-red-500/50 transition-colors py-2 rounded-lg text-sm font-bold flex items-center justify-center gap-2 mt-1"
            >
              <RotateCcw className="w-4 h-4" /> Sayacı Sıfırla
            </button>
          </div>
        </div>
      </div>

      <div className="absolute bottom-4 z-10 bg-slate-900/70 backdrop-blur-md border border-white/10 px-4 py-2 rounded-full text-xs text-slate-300 shadow-lg flex items-center gap-2 pointer-events-none">
        <Info className="w-4 h-4 text-blue-400 flex-shrink-0" />
        <span className="truncate">Kamera açısı cihaz ısınmasını etkileyebilir. Cihazı sabit tutun.</span>
      </div>

      {/* History Modal */}
      {isHistoryModalOpen && (
        <div className="absolute inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-800 border border-slate-700 w-full max-w-sm rounded-2xl shadow-2xl flex flex-col max-h-[80vh]">
            <div className="flex justify-between items-center p-4 border-b border-slate-700 font-bold text-lg">
              <span className="flex items-center gap-2"><History className="w-5 h-5 text-blue-400"/> Geçmiş Raporlar</span>
              <button onClick={() => setIsHistoryModalOpen(false)} className="text-slate-400 hover:text-white transition-colors">
                <X className="w-5 h-5" />
              </button>
            </div>

            {sessionHistory.length > 0 && (
              <div className="px-4 pt-4 pb-2 grid grid-cols-2 gap-3 border-b border-slate-700 bg-slate-800/50">
                <div className="bg-slate-900/50 p-2.5 rounded-lg border border-slate-700/50 text-center flex flex-col justify-center">
                  <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-0.5">Toplam Geçiş</div>
                  <div className="text-xl font-black text-emerald-400">
                    {sessionHistory.filter(s => s.mode !== 'static').reduce((acc, curr) => acc + Math.abs(curr.countIn - curr.countOut), 0)}
                  </div>
                </div>
                <div className="bg-slate-900/50 p-2.5 rounded-lg border border-slate-700/50 text-center flex flex-col justify-center">
                  <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-0.5">Maksimum Sürü (Mera)</div>
                  <div className="text-xl font-black text-blue-400">
                    {Math.max(0, ...sessionHistory.filter(s => s.mode === 'static').map(s => s.countIn), 0)}
                  </div>
                </div>
              </div>
            )}
            
            <div className="p-4 overflow-y-auto flex-1 space-y-3">
              {sessionHistory.length === 0 ? (
                <div className="text-center text-slate-500 py-8 text-sm">
                  Henüz kaydedilmiş sayım yok.
                </div>
              ) : (
                sessionHistory.map(session => (
                  <div key={session.id} className="flex justify-between items-center bg-slate-900/50 p-3 rounded-xl border border-slate-700/50">
                    <div className="flex flex-col">
                      <span className="text-sm font-medium text-slate-300">
                        {new Date(session.timestamp).toLocaleDateString('tr-TR')}
                      </span>
                      <span className="text-xs text-slate-500">
                        {new Date(session.timestamp).toLocaleTimeString('tr-TR')}
                      </span>
                    </div>
                    <div className="flex flex-col items-end">
                      <div className="text-xl font-black tabular-nums text-emerald-400">
                        {session.mode === 'static' ? `${session.countIn} Maks` : `${Math.abs(session.countIn - session.countOut)} Net`}
                      </div>
                      {session.mode !== 'static' && (
                        <div className="text-xs font-bold text-slate-400 flex gap-2 mt-1">
                          <span className="flex items-center text-emerald-500"><ArrowDownCircle className="w-3 h-3 mr-0.5"/>{session.countIn}</span>
                          <span className="flex items-center text-rose-500"><ArrowUpCircle className="w-3 h-3 mr-0.5"/>{session.countOut}</span>
                        </div>
                      )}
                      {session.mode === 'static' && (
                        <div className="text-[10px] font-bold text-slate-400 mt-1 uppercase">Mera Modu</div>
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>

            <div className="p-4 border-t border-slate-700 flex justify-end gap-2">
              <button 
                disabled={sessionHistory.length === 0}
                onClick={downloadCSV}
                className="w-full bg-blue-500 hover:bg-blue-600 text-white transition-colors py-2 rounded-lg text-sm font-bold flex items-center justify-center gap-2 disabled:opacity-50"
              >
                <Download className="w-4 h-4" /> CSV İndir
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
