/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef } from 'react';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import '@tensorflow/tfjs';
import { Camera, Cpu, Info, Loader2, RotateCcw } from 'lucide-react';

interface Track {
  id: number;
  cx: number;
  cy: number;
  box: [number, number, number, number];
  counted: boolean;
  historyX: number;
  historyY: number;
  className: string;
}

export default function App() {
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<string>('');
  const [statusText, setStatusText] = useState('Yapay Zeka Yükleniyor...');
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  const [linePercent, setLinePercent] = useState(50);
  const [isHorizontal, setIsHorizontal] = useState(true);
  const [count, setCount] = useState(0);
  const [countAnimation, setCountAnimation] = useState(false);
  
  // Use refs for moving parts in the animation loop
  const countRef = useRef(count);
  const linePercentRef = useRef(linePercent);
  const isHorizontalRef = useRef(isHorizontal);
  const tracksRef = useRef<Track[]>([]);
  const nextTrackIdRef = useRef(1);
  const animationIdRef = useRef<number | null>(null);
  
  useEffect(() => {
    countRef.current = count;
    linePercentRef.current = linePercent;
    isHorizontalRef.current = isHorizontal;
  }, [count, linePercent, isHorizontal]);

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
    if (!selectedCamera || !model) return;
    
    let stream: MediaStream | null = null;
    
    const startVideo = async () => {
      if (animationIdRef.current) cancelAnimationFrame(animationIdRef.current);
      
      const constraints = {
        video: {
          deviceId: { exact: selectedCamera },
          width: { ideal: 640 },
          height: { ideal: 480 },
        }
      };

      try {
        stream = await navigator.mediaDevices.getUserMedia(constraints);
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            if (canvasRef.current && videoRef.current) {
              canvasRef.current.width = videoRef.current.videoWidth;
              canvasRef.current.height = videoRef.current.videoHeight;
              detectFrame();
            }
          };
        }
      } catch (err) {
        console.error("Kamera başlatılamadı:", err);
      }
    };

    startVideo();

    return () => {
      if (stream) stream.getTracks().forEach(t => t.stop());
      if (animationIdRef.current) cancelAnimationFrame(animationIdRef.current);
    };
  }, [selectedCamera, model]);

  const detectFrame = async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.paused || video.ended || !model) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const isHoriz = isHorizontalRef.current;
    const lPercent = linePercentRef.current / 100;
    const linePos = isHoriz ? (canvas.height * lPercent) : (canvas.width * lPercent);

    if (video.videoWidth === 0 || video.videoHeight === 0) {
      animationIdRef.current = requestAnimationFrame(detectFrame);
      return;
    }

    try {
      const predictions = await model.detect(video);
      const allowedClasses = ['sheep', 'cow', 'horse', 'person', 'dog', 'cat'];
      const animalDetections = predictions.filter(p => allowedClasses.includes(p.class));

      updateTracker(animalDetections, linePos, isHoriz);
    } catch (e) {
      console.error("Detection error", e);
    }

    // Draw Line
    ctx.beginPath();
    if (isHoriz) {
      ctx.moveTo(0, linePos);
      ctx.lineTo(canvas.width, linePos);
    } else {
      ctx.moveTo(linePos, 0);
      ctx.lineTo(linePos, canvas.height);
    }
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 4;
    ctx.setLineDash([10, 10]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw Boxes
    for (let track of tracksRef.current) {
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

    animationIdRef.current = requestAnimationFrame(detectFrame);
  };

  const updateTracker = (detections: cocoSsd.DetectedObject[], linePos: number, isHoriz: boolean) => {
    let currentTracks: Track[] = [];
    let tracks = tracksRef.current;
    const MAX_DIST = 300;
    
    for (let det of detections) {
      let cx = det.bbox[0] + (det.bbox[2] / 2);
      let cy = det.bbox[1] + (det.bbox[3] / 2);

      let bestMatch: Track | null = null;
      let min_dist = MAX_DIST;

      for (let t of tracks) {
        let dist = Math.hypot(t.cx - cx, t.cy - cy);
        if (dist < min_dist) {
          min_dist = dist;
          bestMatch = t;
        }
      }

      if (bestMatch) {
        currentTracks.push({
          id: bestMatch.id,
          cx: cx,
          cy: cy,
          box: det.bbox as [number, number, number, number],
          counted: bestMatch.counted,
          historyX: bestMatch.cx,
          historyY: bestMatch.cy,
          className: det.class
        });
        tracks = tracks.filter(t => t.id !== bestMatch!.id);
      } else {
        currentTracks.push({
          id: nextTrackIdRef.current++,
          cx: cx,
          cy: cy,
          box: det.bbox as [number, number, number, number],
          counted: false,
          historyX: cx,
          historyY: cy,
          className: det.class
        });
      }
    }

    for (let t of currentTracks) {
      if (!t.counted) {
        let crossed = false;
        
        if (isHoriz) {
          if ((t.historyY <= linePos && t.cy >= linePos) || (t.historyY >= linePos && t.cy <= linePos)) {
            crossed = true;
          }
        } else {
          if ((t.historyX <= linePos && t.cx >= linePos) || (t.historyX >= linePos && t.cx <= linePos)) {
            crossed = true;
          }
        }

        if (crossed) {
          setCount(prev => prev + 1);
          setCountAnimation(true);
          setTimeout(() => setCountAnimation(false), 200);
          t.counted = true;
        }
      }
    }

    tracksRef.current = currentTracks;
  };

  const resetCounter = () => {
    setCount(0);
    tracksRef.current = [];
  };

  return (
    <div className="bg-slate-900 text-white h-screen w-screen relative flex flex-col items-center justify-center overflow-hidden font-sans">
      <div className="absolute inset-0 z-0 flex items-center justify-center bg-black">
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
        
        <div className="bg-slate-900/70 backdrop-blur-md border border-white/10 px-6 py-4 rounded-2xl shadow-2xl pointer-events-auto flex flex-col items-center self-start">
          <span className="text-slate-300 text-xs font-bold uppercase tracking-widest mb-1">Toplam Sayım</span>
          <span className={`text-5xl font-extrabold tabular-nums drop-shadow-md transition-all duration-200 ${countAnimation ? 'scale-125 text-emerald-400' : 'text-white scale-100'}`}>
            {count}
          </span>
        </div>

        <div className="bg-slate-900/70 backdrop-blur-md border border-white/10 p-4 rounded-2xl shadow-2xl pointer-events-auto w-full max-w-xs space-y-4">
          <div className={`flex items-center gap-3 text-sm font-medium ${model ? 'text-emerald-400' : 'text-blue-400'}`}>
            {!model ? <Loader2 className="w-5 h-5 animate-spin" /> : <Cpu className="w-5 h-5" />}
            <span>{statusText}</span>
          </div>

          <div className="h-px bg-slate-700 w-full" />

          <div className="space-y-1">
            <label className="text-xs text-slate-400 font-semibold">Kamera Kaynağı</label>
            <div className="relative">
              <select 
                value={selectedCamera}
                onChange={e => setSelectedCamera(e.target.value)}
                disabled={cameras.length === 0}
                className="w-full bg-slate-800 border border-slate-600 text-white text-sm rounded-lg p-2.5 appearance-none focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
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
          </div>

          <div className="space-y-1">
            <label className="text-xs text-slate-400 font-semibold">Çizgi Yönü</label>
            <select
              value={isHorizontal ? 'horizontal' : 'vertical'}
              onChange={e => setIsHorizontal(e.target.value === 'horizontal')}
              className="w-full bg-slate-800 border border-slate-600 text-white text-sm rounded-lg p-2.5 focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer"
            >
              <option value="horizontal">Yatay (Aşağı/Yukarı Geçiş)</option>
              <option value="vertical">Dikey (Sağ/Sol Geçiş)</option>
            </select>
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

          <button 
            onClick={resetCounter}
            className="w-full bg-red-500/20 hover:bg-red-500/30 text-red-500 border border-red-500/50 transition-colors py-2 rounded-lg text-sm font-bold flex items-center justify-center gap-2"
          >
            <RotateCcw className="w-4 h-4" /> Sayacı Sıfırla
          </button>
        </div>
      </div>

      <div className="absolute bottom-4 z-10 bg-slate-900/70 backdrop-blur-md border border-white/10 px-4 py-2 rounded-full text-xs text-slate-300 shadow-lg flex items-center gap-2 pointer-events-none">
        <Info className="w-4 h-4 text-blue-400 flex-shrink-0" />
        <span className="truncate">Kamera açısı cihaz ısınmasını etkileyebilir. Cihazı sabit tutun.</span>
      </div>
    </div>
  );
}
