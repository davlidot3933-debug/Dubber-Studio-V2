import { GoogleGenAI, Modality } from '@google/genai';
import { Film, FileText, Play, Pause, Languages, Loader2, Music, Download, Settings, Square, ListChecks, X, Undo2, Redo2, Link2, Link2Off, Plus, Minus, ZoomIn, ZoomOut, HelpCircle } from 'lucide-react';
import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import { motion } from 'motion/react';

// Initialize Gemini API
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

const TTS_VOICES = [
  { id: 'Puck', label: 'Male: Puck' },
  { id: 'Charon', label: 'Male: Charon' },
  { id: 'Kore', label: 'Female: Kore' },
  { id: 'Fenrir', label: 'Male: Fenrir' },
  { id: 'Aoede', label: 'Female: Aoede' }
];

const VOXCPM_VOICES = [
  { id: 'alloy', label: 'Neutral: Alloy' },
  { id: 'echo', label: 'Male: Echo' },
  { id: 'fable', label: 'British Male: Fable' },
  { id: 'onyx', label: 'Deep Male: Onyx' },
  { id: 'nova', label: 'Female: Nova' },
  { id: 'shimmer', label: 'Clear Female: Shimmer' }
];

interface Subtitle {
  id: number;
  startTime: number;
  endTime: number;
  text: string;
  audioUrl?: string; // object URL to the generated WAV
  audioDuration?: number; // duration of the generated audio in seconds
  audioTrimStart?: number; // offset from beginning (seconds)
  audioTrimEnd?: number; // point where it stops (seconds)
  isGenerating?: boolean;
  voice: string;
  engine?: string;
  refAudioFile?: File;
  refAudioBase64?: string;
  audioStartTime?: number;
  isLinked?: boolean;
  waveform?: number[];
}

// Convert "00:00:01,000" to seconds
function parseTime(timeStr: string): number {
  const parts = timeStr.trim().split(',');
  const hms = parts[0].split(':');
  const ms = parts[1] ? Number(parts[1]) : 0;
  return Number(hms[0]) * 3600 + Number(hms[1]) * 60 + Number(hms[2]) + ms / 1000;
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

function getAudioDuration(url: string): Promise<number> {
  return new Promise((resolve) => {
    const audio = new Audio(url);
    audio.onloadedmetadata = () => {
      resolve(audio.duration);
    };
    audio.onerror = () => {
      resolve(0);
    };
  });
}

async function generateWaveform(url: string, numberOfSamples: number = 100): Promise<number[]> {
  try {
    const response = await fetch(url);
    const arrayBuffer = await response.arrayBuffer();
    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    const channelData = audioBuffer.getChannelData(0);
    
    const samples = [];
    const blockSize = Math.floor(channelData.length / numberOfSamples);
    
    for (let i = 0; i < numberOfSamples; i++) {
        let sum = 0;
        for (let j = 0; j < blockSize; j++) {
            const val = channelData[Math.min(channelData.length - 1, i * blockSize + j)];
            sum += val * val;
        }
        samples.push(Math.sqrt(sum / blockSize));
    }
    
    // Close context to free resources
    await audioContext.close();
    
    // Normalize samples with a bit of "boost" for visual appeal
    const max = Math.max(...samples) || 1;
    return samples.map(s => Math.pow(s / max, 0.8)); // Power curve to emphasize quieter parts
  } catch (e) {
    console.warn('Failed to generate waveform', e);
    return [];
  }
}

// Parse standard SRT format
function parseSRT(srt: string): Subtitle[] {
  const normalized = srt.replace(/\r\n/g, '\n');
  const blocks = normalized.split('\n\n');
  const subtitles: Subtitle[] = [];

  for (const block of blocks) {
    const lines = block.split('\n').map((l) => l.trim()).filter((l) => l.length > 0);
    if (lines.length >= 3) {
      const id = parseInt(lines[0], 10);
      const timeMatch = lines[1].match(/(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})/);
      if (timeMatch) {
        const startTime = parseTime(timeMatch[1]);
        const endTime = parseTime(timeMatch[2]);
        const text = lines.slice(2).join(' ').trim();
        if (text) {
          subtitles.push({ id, startTime, endTime, text, voice: 'default', audioStartTime: startTime, isLinked: true });
        }
      }
    }
  }
  return subtitles;
}

// Convert PCM16 to WAV
function encodeWAV(samples: Int16Array, sampleRate: number = 24000): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  const writeString = (view: DataView, offset: number, string: string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); // Mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    view.setInt16(offset, samples[i], true);
  }

  return new Blob([view], { type: 'audio/wav' });
}

// Memoized Timeline Item for better performance
interface TimelineSubtitleItemProps {
  sub: Subtitle;
  currentTime: number;
  totalDuration: number;
  zoomLevel: number;
  draggingSub: { id: number; offset: number; type: string } | null;
  handleTimelineDrag: (subId: number, deltaX: number, type: 'text' | 'audio' | 'trim-audio-start' | 'trim-audio-end' | 'trim-text-start' | 'trim-text-end') => void;
  handleTimelineDragEnd: (subId: number, info: any, type: 'text' | 'audio' | 'trim-audio-start' | 'trim-audio-end' | 'trim-text-start' | 'trim-text-end') => void;
  handleToggleLink: (id: number) => void;
  videoRef: React.RefObject<HTMLVideoElement>;
  isActive: boolean;
  isOverlapping?: boolean;
}

const TimelineSubtitleItem = React.memo(({ 
  sub, 
  currentTime, 
  totalDuration, 
  zoomLevel, 
  draggingSub, 
  handleTimelineDrag, 
  handleTimelineDragEnd, 
  handleToggleLink,
  videoRef,
  isActive,
  isOverlapping
}: TimelineSubtitleItemProps) => {
  const audioDur = Math.max(0.01, (sub.audioTrimEnd ?? sub.audioDuration ?? (sub.endTime - sub.startTime)) - (sub.audioTrimStart ?? 0));
  const safeTotalDuration = totalDuration > 0 && Number.isFinite(totalDuration) ? totalDuration : 1;
  const leftPerc = (sub.startTime / safeTotalDuration) * 100;
  const widthPerc = Math.max(0.01, (sub.endTime - sub.startTime) / safeTotalDuration) * 100;
  const isLinked = sub.isLinked ?? true;
  
  const isDraggingThisText = draggingSub?.id === sub.id && draggingSub?.type === 'text';
  const isDraggingThisAudio = draggingSub?.id === sub.id && draggingSub?.type === 'audio';
  const isTrimmingTextStart = draggingSub?.id === sub.id && draggingSub?.type === 'trim-text-start';
  const isTrimmingTextEnd = draggingSub?.id === sub.id && draggingSub?.type === 'trim-text-end';
  const isTrimmingAudioStart = draggingSub?.id === sub.id && draggingSub?.type === 'trim-audio-start';
  const isTrimmingAudioEnd = draggingSub?.id === sub.id && draggingSub?.type === 'trim-audio-end';
  
  const dragDeltaTime = (draggingSub?.id === sub.id && zoomLevel > 0) ? (draggingSub.offset / zoomLevel) : 0;
  
  // Real-time calculated positions for visual feedback
  let displayStartTime = sub.startTime;
  let displayEndTime = sub.endTime;
  let displayAudioStartTime = sub.audioStartTime ?? sub.startTime;
  let displayAudioTrimStart = sub.audioTrimStart ?? 0;
  let displayAudioTrimEnd = sub.audioTrimEnd ?? sub.audioDuration ?? (sub.endTime - sub.startTime);

  if (isDraggingThisText) {
    displayStartTime += dragDeltaTime;
    displayEndTime += dragDeltaTime;
    if (isLinked) displayAudioStartTime += dragDeltaTime;
  } else if (isDraggingThisAudio) {
    displayAudioStartTime += dragDeltaTime;
    if (isLinked) {
      displayStartTime += dragDeltaTime;
      displayEndTime += dragDeltaTime;
    }
  } else if (isTrimmingTextStart) {
    displayStartTime = Math.max(0, Math.min(sub.endTime - 0.1, sub.startTime + dragDeltaTime));
    if (isLinked) displayAudioStartTime = displayStartTime;
  } else if (isTrimmingTextEnd) {
    displayEndTime = Math.max(sub.startTime + 0.1, sub.endTime + dragDeltaTime);
    if (isLinked) {
      // If linked, trimming text end usually doesn't affect audio unless we want it to
    }
  } else if (isTrimmingAudioStart) {
    const audioDur = sub.audioDuration || (sub.endTime - sub.startTime);
    const audioTrimEnd = sub.audioTrimEnd ?? audioDur;
    displayAudioTrimStart = Math.max(0, Math.min(audioTrimEnd - 0.1, (sub.audioTrimStart ?? 0) + dragDeltaTime));
    const actualDelta = displayAudioTrimStart - (sub.audioTrimStart ?? 0);
    displayAudioStartTime = (sub.audioStartTime ?? sub.startTime) + actualDelta;
    if (isLinked) displayStartTime = displayAudioStartTime;
  } else if (isTrimmingAudioEnd) {
    const audioDur = sub.audioDuration || (sub.endTime - sub.startTime);
    displayAudioTrimEnd = Math.max((sub.audioTrimStart ?? 0) + 0.1, Math.min(audioDur, (sub.audioTrimEnd ?? audioDur) + dragDeltaTime));
    if (isLinked) displayEndTime = displayAudioStartTime + (displayAudioTrimEnd - displayAudioTrimStart);
  }

  const displayLeftPerc = (displayStartTime / safeTotalDuration) * 100;
  const displayWidthPerc = Math.max(0.01, (displayEndTime - displayStartTime) / safeTotalDuration) * 100;
  const displayAudioLeftPerc = (displayAudioStartTime / safeTotalDuration) * 100;
  const displayAudioWidthPerc = ((displayAudioTrimEnd - displayAudioTrimStart) / safeTotalDuration) * 100;

  const playProgress = audioDur > 0 ? (currentTime - displayAudioStartTime) / audioDur : 0;
  const safePlayProgress = Math.max(0, Math.min(1, Number.isFinite(playProgress) ? playProgress : 0));
  const playheadX = safePlayProgress * (sub.waveform?.length || 0);

  // Voice color mapping
  const getTrackColor = () => {
    if (isOverlapping) return 'bg-rose-500/30 border-rose-400 ring-rose-500/50';
    if (sub.engine === 'google') return 'bg-emerald-500/20 border-emerald-500/60 text-emerald-300';
    if (sub.engine === 'voxcpm') {
      const colors: Record<string, string> = {
        alloy: 'bg-blue-500/20 border-blue-500/60 text-blue-300',
        echo: 'bg-indigo-500/20 border-indigo-500/60 text-indigo-300',
        fable: 'bg-purple-500/20 border-purple-500/60 text-purple-300',
        onyx: 'bg-slate-500/20 border-slate-500/60 text-slate-300',
        nova: 'bg-pink-500/20 border-pink-500/60 text-pink-300',
        shimmer: 'bg-orange-500/20 border-orange-500/60 text-orange-300'
      };
      return colors[sub.voice] || 'bg-violet-500/20 border-violet-500/60 text-violet-300';
    }
    return 'bg-amber-500/20 border-amber-500/60 text-amber-300';
  };

  const trackStyles = getTrackColor();

  return (
    <div className="absolute inset-x-0 top-0 h-full pointer-events-none group/item">
      {/* Connector line between Text and Audio if unlinked and both visible */}
      {!isLinked && sub.audioUrl && (
        <svg className="absolute inset-0 w-full h-full pointer-events-none opacity-20">
          <line 
            x1={`${displayLeftPerc + displayWidthPerc/2}%`} 
            y1="40" 
            x2={`${displayAudioLeftPerc + displayAudioWidthPerc/2}%`} 
            y2="60" 
            stroke="currentColor" 
            strokeWidth="1" 
            strokeDasharray="4 2"
            className="text-slate-500"
          />
        </svg>
      )}

      {/* Subtitle Text Track */}
      <motion.div 
        drag="x"
        dragMomentum={false}
        dragElastic={0}
        onDrag={(e, info) => handleTimelineDrag(sub.id, info.offset.x, 'text')}
        onDragEnd={(e, info) => handleTimelineDragEnd(sub.id, info, 'text')}
        className={`absolute top-1 h-10 rounded px-1 flex items-center border cursor-pointer group/sub transition-shadow pointer-events-auto ${
          isActive 
            ? (sub.audioUrl ? 'bg-amber-400 border-amber-300 text-amber-950 shadow-[0_0_15px_rgba(251,191,36,0.6)] z-30 scale-y-110 origin-bottom ring-2 ring-amber-400/50' : 'bg-amber-500/20 border-amber-400/80 text-amber-200 shadow-[0_0_8px_rgba(245,158,11,0.3)] z-20 scale-y-105')
            : (sub.audioUrl 
                ? 'bg-emerald-500/20 border-emerald-500/60 text-emerald-300 hover:border-emerald-400 z-10 hover:z-20' 
                : 'bg-slate-800/50 border-slate-700 text-slate-400 hover:border-slate-500 z-0 hover:z-10')
        } ${isLinked ? 'border-solid' : 'border-dashed border-slate-500/50'}`}
        style={{ 
          left: `${displayLeftPerc}%`, 
          width: `${displayWidthPerc}%`, 
          minWidth: '4px'
        }}
        onPointerDown={(e) => {
          e.stopPropagation();
          if (videoRef.current && Number.isFinite(displayStartTime)) {
            videoRef.current.currentTime = displayStartTime;
          }
        }}
      >
        {/* Link Toggle Button */}
        {sub.audioUrl && (
          <button 
            onClick={(e) => { e.stopPropagation(); handleToggleLink(sub.id); }}
            className={`absolute -top-6 left-1/2 -translate-x-1/2 p-1 rounded-full bg-slate-800 border border-slate-700 shadow-lg opacity-0 group-hover/sub:opacity-100 transition-opacity z-50 pointer-events-auto ${isLinked ? 'text-amber-400 border-amber-500/50' : 'text-slate-400'}`}
            title={isLinked ? "Unlink Audio" : "Link Audio"}
          >
            {isLinked ? <Link2 className="w-3 h-3" /> : <Link2Off className="w-3 h-3" />}
          </button>
        )}
        
        {/* Text Trim Info Tooltip */}
        {(isTrimmingTextStart || isTrimmingTextEnd) && (
          <div 
            className="absolute -top-12 z-[60] pointer-events-none flex flex-col items-center"
            style={{ 
              left: isTrimmingTextStart ? '0%' : '100%',
              transform: 'translateX(-50%)' 
            }}
          >
            <div className="bg-slate-900 border border-amber-500 rounded px-2 py-1 shadow-[0_0_15px_rgba(245,158,11,0.3)] flex flex-col items-center min-w-[60px]">
              <div className="text-[8px] text-amber-500 font-bold uppercase tracking-wider leading-none mb-1">Duration</div>
              <div className="text-xs text-white font-mono font-bold leading-none">
                {(displayEndTime - displayStartTime).toFixed(2)}s
              </div>
              <div className="absolute -bottom-1 left-1/2 -translate-x-1/2 w-2 h-2 bg-slate-900 border-r border-b border-amber-500 rotate-45" />
            </div>
          </div>
        )}
        
        {/* Left Trim Handle (Text) */}
        <div 
          className="absolute left-0 top-0 bottom-0 w-3 hover:bg-white/20 cursor-ew-resize z-40 transition-colors flex items-center justify-center group/trim-text-l text-white pointer-events-auto shadow-inner"
          onPointerDown={(e) => e.stopPropagation()}
        >
          <motion.div
            drag="x"
            dragMomentum={false}
            dragElastic={0}
            onDrag={(e, info) => handleTimelineDrag(sub.id, info.offset.x, 'trim-text-start')}
            onDragEnd={(e, info) => handleTimelineDragEnd(sub.id, info, 'trim-text-start')}
            className={`w-1.5 h-4 rounded-full transition-all ${isTrimmingTextStart ? 'bg-amber-400 scale-x-150' : 'bg-white/30 group-hover/sub:bg-white/60'}`}
          />
        </div>

        {/* Right Trim Handle (Text) */}
        <div 
          className="absolute right-0 top-0 bottom-0 w-3 hover:bg-white/20 cursor-ew-resize z-40 transition-colors flex items-center justify-center group/trim-text-r text-white pointer-events-auto shadow-inner"
          onPointerDown={(e) => e.stopPropagation()}
        >
          <motion.div
            drag="x"
            dragMomentum={false}
            dragElastic={0}
            onDrag={(e, info) => handleTimelineDrag(sub.id, info.offset.x, 'trim-text-end')}
            onDragEnd={(e, info) => handleTimelineDragEnd(sub.id, info, 'trim-text-end')}
            className={`w-1.5 h-4 rounded-full transition-all ${isTrimmingTextEnd ? 'bg-amber-400 scale-x-150' : 'bg-white/30 group-hover/sub:bg-white/60'}`}
          />
        </div>

        <div className="absolute inset-0 overflow-hidden flex items-center justify-center px-1 pointer-events-none">
          <span className="text-[10px] whitespace-nowrap truncate font-medium drop-shadow-md pb-0.5 w-full text-center">
             {sub.text}
          </span>
        </div>
      </motion.div>

      {/* Audio Clip Track */}
      {sub.audioUrl && (
        <div 
          className="absolute inset-x-0 pointer-events-none" 
          style={{ top: '48px', height: '40px' }}
        >
          <motion.div 
            drag="x"
            dragMomentum={false}
            dragElastic={0}
            onDrag={(e, info) => handleTimelineDrag(sub.id, info.offset.x, 'audio')}
            onDragEnd={(e, info) => handleTimelineDragEnd(sub.id, info, 'audio')}
            className={`absolute top-1 bottom-1 rounded flex items-center border cursor-pointer overflow-hidden group/audio transition-all duration-300 pointer-events-auto ${
              isActive
                ? `${isOverlapping ? 'bg-rose-950 border-rose-400 shadow-[0_0_15px_rgba(244,63,94,0.5)] hover:shadow-[0_0_25px_rgba(244,63,94,0.7)]' : 'bg-purple-900 border-purple-400 shadow-[0_0_15px_rgba(168,85,247,0.5)] hover:shadow-[0_0_25px_rgba(168,85,247,0.85)]'} z-40 scale-y-110 origin-top ring-1 ring-white/30`
                : `${trackStyles} z-10 hover:border-white/60 hover:brightness-110 hover:shadow-lg ring-0`
            } ${isLinked ? 'border-solid' : 'border-dashed'}`}
            style={{ 
              left: `${displayAudioLeftPerc}%`, 
              width: `${displayAudioWidthPerc}%`, 
              minWidth: '10px'
            }}
          >
            {/* Overlap Warning Indicator */}
            {isOverlapping && (
              <div className="absolute inset-0 bg-[repeating-linear-gradient(45deg,transparent,transparent_10px,rgba(244,63,94,0.15)_10px,rgba(244,63,94,0.15)_20px)] pointer-events-none animate-pulse z-0" />
            )}
            
            {/* Waveform Visualization - Optimized with single path */}
            {sub.waveform && sub.waveform.length > 0 && (
              <div className="absolute inset-0 w-full h-full px-1 py-1.5 overflow-hidden pointer-events-none">
                <svg className="w-full h-full" preserveAspectRatio="none" viewBox={`0 0 ${sub.waveform.length} 100`}>
                  <defs>
                    <clipPath id={`wf-clip-${sub.id}`}>
                      <rect 
                        x="0" 
                        y="0" 
                        width={playheadX} 
                        height="100" 
                      />
                    </clipPath>
                  </defs>
                  
                  {/* Optimized path data for waveform */}
                  {(() => {
                    let d = "";
                    for (let i = 0; i < sub.waveform.length; i++) {
                      const val = Number.isFinite(sub.waveform[i]) ? sub.waveform[i] : 0;
                      const h = val * 90;
                      const y1 = 50 - h/2;
                      const y2 = 50 + h/2;
                      d += `M${i},${y1} L${i},${y2} `;
                    }
                    
                    return (
                      <>
                        {/* Background Waveform (Unplayed) */}
                        <path 
                          d={d}
                          strokeWidth="0.6"
                          stroke="currentColor"
                          strokeLinecap="round"
                          className={isActive ? "text-purple-100/20" : "text-indigo-100/10"}
                        />
                        
                        {/* Foreground Waveform (Played) */}
                        <path 
                          clipPath={`url(#wf-clip-${sub.id})`}
                          d={d}
                          strokeWidth="0.6"
                          stroke="currentColor"
                          strokeLinecap="round"
                          className={`transition-all duration-300 ${isActive ? "text-amber-400 drop-shadow-[0_0_2px_rgba(251,191,36,0.5)] group-hover/audio:drop-shadow-[0_0_8px_rgba(251,191,36,0.9)] group-hover/audio:brightness-125" : "text-emerald-400 drop-shadow-[0_0_2px_rgba(52,211,153,0.5)] group-hover/audio:brightness-150"}`}
                        />
                      </>
                    );
                  })()}

                  {/* Audio Playhead Indicator (Internal) */}
                  {currentTime >= displayAudioStartTime && currentTime <= displayAudioStartTime + audioDur && (
                    <rect 
                      x={playheadX} 
                      y="10" 
                      width={0.8} 
                      height="80" 
                      fill="white"
                      className="opacity-80"
                      rx={0.4}
                    />
                  )}
                </svg>
              </div>
            )}
            
            {!sub.waveform && (
              <div className="absolute inset-0 flex items-center justify-center opacity-20 pointer-events-none">
                 <div className="w-full h-1 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-pulse" />
              </div>
            )}

            {/* Trim Info Tooltip */}
            {(isTrimmingAudioStart || isTrimmingAudioEnd) && (
              <div 
                className="absolute -top-14 z-[60] pointer-events-none flex flex-col items-center"
                style={{ 
                  left: isTrimmingAudioStart ? '0%' : '100%',
                  transform: 'translateX(-50%)' 
                }}
              >
                <div className="bg-slate-900 border border-purple-500 rounded px-2.5 py-1.5 shadow-[0_0_20px_rgba(168,85,247,0.4)] flex flex-col items-center min-w-[70px]">
                  <div className="text-[9px] text-purple-400 font-bold uppercase tracking-wider leading-none mb-1">Duration</div>
                  <div className="text-sm text-white font-mono font-bold leading-none">
                    {(displayAudioTrimEnd - displayAudioTrimStart).toFixed(2)}s
                  </div>
                  <div className="absolute -bottom-1 left-1/2 -translate-x-1/2 w-2 h-2 bg-slate-900 border-r border-b border-purple-500 rotate-45" />
                </div>
              </div>
            )}

            {/* Audio Trim Handles - Enhanced Branding */}
            <div 
              className="absolute left-0 top-0 bottom-0 w-4 hover:bg-white/10 cursor-ew-resize z-50 transition-colors flex items-center justify-start group/trim-audio-l pointer-events-auto"
              onPointerDown={(e) => e.stopPropagation()}
            >
              <div className="h-full w-1 bg-purple-400 group-hover/audio:bg-white transition-colors" />
              <motion.div
                drag="x"
                dragMomentum={false}
                dragElastic={0}
                onDrag={(e, info) => handleTimelineDrag(sub.id, info.offset.x, 'trim-audio-start')}
                onDragEnd={(e, info) => handleTimelineDragEnd(sub.id, info, 'trim-audio-start')}
                className={`w-2 h-7 -ml-0.5 rounded-r bg-purple-400/80 flex items-center justify-center transition-all ${isTrimmingAudioStart ? 'scale-x-125 bg-amber-400 shadow-lg' : 'opacity-0 group-hover/trim-audio-l:opacity-100'}`}
              >
                <div className="w-[1px] h-3 bg-white/50" />
              </motion.div>
            </div>
            
            <div 
              className="absolute right-0 top-0 bottom-0 w-4 hover:bg-white/10 cursor-ew-resize z-50 transition-colors flex items-center justify-end group/trim-audio-r pointer-events-auto"
              onPointerDown={(e) => e.stopPropagation()}
            >
              <motion.div
                drag="x"
                dragMomentum={false}
                dragElastic={0}
                onDrag={(e, info) => handleTimelineDrag(sub.id, info.offset.x, 'trim-audio-end')}
                onDragEnd={(e, info) => handleTimelineDragEnd(sub.id, info, 'trim-audio-end')}
                className={`w-2 h-7 -mr-0.5 rounded-l bg-purple-400/80 flex items-center justify-center transition-all ${isTrimmingAudioEnd ? 'scale-x-125 bg-amber-400 shadow-lg' : 'opacity-0 group-hover/trim-audio-r:opacity-100'}`}
              >
                <div className="w-[1px] h-3 bg-white/50" />
              </motion.div>
              <div className="h-full w-1 bg-purple-400 group-hover/audio:bg-white transition-colors" />
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
}, (prev, next) => {
  // If this item is being dragged now or was being dragged, re-render
  if (prev.draggingSub?.id === prev.sub.id || next.draggingSub?.id === next.sub.id) {
    return false;
  }
  
  // If the active state changed, must re-render
  if (prev.isActive !== next.isActive) return false;

  // Standard memoization for structural changes
  const structuralCheck = 
    prev.sub === next.sub &&
    prev.totalDuration === next.totalDuration &&
    prev.zoomLevel === next.zoomLevel &&
    prev.isOverlapping === next.isOverlapping;

  if (!structuralCheck) return false;

  // If it's active, we need re-renders for the internal playhead
  if (next.isActive) {
     return prev.currentTime === next.currentTime;
  }

  // If it's inactive, we can skip the currentTime update
  return true;
});

// Memoized Sidebar List Item for better performance
interface SubtitleListItemProps {
  sub: Subtitle;
  isActive: boolean;
  isBatchEditMode: boolean;
  selectedSubtitles: Set<number>;
  previewingId: number | null;
  isGeneratingAll: boolean;
  ttsEngine: string;
  defaultVoxCPMVoice: string;
  referenceAudioFile: File | null;
  referenceAudioBase64: string | null;
  handleToggleLink: (id: number) => void;
  toggleSubtitleSelection: (id: number) => void;
  handlePreviewAudio: (e: React.MouseEvent, sub: Subtitle) => void;
  handleGenerateSingle: (e: React.MouseEvent, sub: Subtitle) => void;
  handleEngineChange: (id: number, engine: string) => void;
  handleVoiceChange: (id: number, voice: string) => void;
  handleSubReferenceAudioUpload: (id: number, e: React.ChangeEvent<HTMLInputElement>) => void;
  playAudioFile: (file: File) => void;
  updateSubtitles: (updater: Subtitle[] | ((prev: Subtitle[]) => Subtitle[]), skipHistory?: boolean) => void;
  videoRef: React.RefObject<HTMLVideoElement>;
}

const SubtitleListItem = React.memo(({ 
  sub, 
  isActive, 
  isBatchEditMode, 
  selectedSubtitles, 
  previewingId,
  isGeneratingAll,
  ttsEngine,
  defaultVoxCPMVoice,
  referenceAudioFile,
  referenceAudioBase64,
  handleToggleLink,
  toggleSubtitleSelection,
  handlePreviewAudio,
  handleGenerateSingle,
  handleEngineChange,
  handleVoiceChange,
  handleSubReferenceAudioUpload,
  playAudioFile,
  updateSubtitles,
  videoRef
}: SubtitleListItemProps) => {
  return (
    <div 
      className={`p-3 border-b border-slate-800/50 transition-colors cursor-pointer ${isActive && !isBatchEditMode ? 'bg-slate-800/30 border-l-2 border-l-amber-500' : 'hover:bg-slate-800/20'} ${selectedSubtitles.has(sub.id) ? 'bg-amber-900/20 border-l-2 border-l-amber-500' : ''}`}
      onClick={() => {
        if (isBatchEditMode) {
          toggleSubtitleSelection(sub.id);
        } else {
          if (videoRef.current) {
            videoRef.current.currentTime = sub.startTime;
          }
        }
      }}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <button 
            onClick={(e) => { e.stopPropagation(); handleToggleLink(sub.id); }}
            className={`p-1 rounded transition ${sub.isLinked === false ? 'bg-rose-500/10 text-rose-500' : 'text-slate-500 hover:text-slate-300'}`}
            title={sub.isLinked === false ? "Unlinked" : "Linked"}
          >
            {sub.isLinked === false ? <Link2Off className="w-3.5 h-3.5" /> : <Link2 className="w-3.5 h-3.5" />}
          </button>
          {isBatchEditMode && (
            <input 
              type="checkbox" 
              checked={selectedSubtitles.has(sub.id)}
              readOnly
              className="w-3 h-3 rounded border-slate-700 bg-slate-900 checked:bg-amber-500"
            />
          )}
          <span className={`text-[10px] ${isActive && !isBatchEditMode ? 'text-amber-500' : 'text-slate-500'}`}>
            {formatTime(sub.startTime)}
          </span>
        </div>
        
        <div className="flex items-center gap-1">
          {sub.isGenerating ? (
            <span className="text-[10px] text-amber-500 flex items-center gap-1">
              <Loader2 className="w-3 h-3 animate-spin"/>
            </span>
          ) : sub.audioUrl ? (
             <div className="flex items-center gap-2">
              {isActive && <span className="text-[10px] text-emerald-400">Playing</span>}
              <button 
                onClick={(e) => handlePreviewAudio(e, sub)}
                className="text-amber-500 hover:text-amber-400 transition-colors p-1 hover:bg-slate-800 rounded"
                title="Preview Audio"
              >
                {previewingId === sub.id ? <Square className="w-3.5 h-3.5 fill-current" /> : <Play className="w-3.5 h-3.5 fill-current" />}
              </button>
              <a href={sub.audioUrl} download={`sub_${sub.id}.wav`} className="text-slate-500 hover:text-white transition-colors p-1 hover:bg-slate-800 rounded" title="Download WAV" onClick={(e) => e.stopPropagation()}>
                <Download className="w-3.5 h-3.5" />
              </a>
            </div>
          ) : (
            <button
                onClick={(e) => handleGenerateSingle(e, sub)}
                disabled={isGeneratingAll}
                className="text-[10px] text-slate-400 hover:text-amber-500 transition-colors px-2 py-1 border border-slate-700 hover:border-amber-500/50 rounded bg-slate-800 disabled:opacity-50"
            >
                Generate
            </button>
          )}
        </div>
      </div>
      <textarea 
        value={sub.text}
        onClick={(e) => e.stopPropagation()}
        onChange={(e) => {
           updateSubtitles(prev => prev.map(s => s.id === sub.id ? { ...s, text: e.target.value } : s));
        }}
        className={`w-full bg-slate-900/50 border border-slate-700/50 rounded-md p-2 text-sm leading-relaxed mb-2 focus:outline-none focus:border-amber-500/50 focus:bg-slate-900 resize-y min-h-[60px] custom-scrollbar transition-colors ${isActive ? 'text-slate-200 font-medium' : 'text-slate-400 focus:text-slate-200'}`}
        placeholder="Subtitle text..."
      />
      <div className="flex items-center justify-end gap-2">
        <select
          value={sub.engine || 'default'}
          onClick={(e) => e.stopPropagation()}
          onChange={(e) => handleEngineChange(sub.id, e.target.value)}
          className="bg-slate-900 text-[10px] text-slate-400 border border-slate-700 rounded px-2 py-1 outline-none hover:border-slate-500/50 transition-colors"
        >
          <option value="default">Global Engine</option>
          <option value="gemini">Gemini</option>
          <option value="voxcpm">VoxCPM</option>
          <option value="google-free">Google Free</option>
        </select>
        {(() => {
          const eng = (!sub.engine || sub.engine === 'default') ? ttsEngine : sub.engine;
          return eng === 'gemini' ? (
          <select 
            value={sub.voice} 
            onClick={(e) => e.stopPropagation()}
            onChange={(e) => handleVoiceChange(sub.id, e.target.value)}
            className="bg-slate-900 text-[10px] text-slate-400 border border-slate-700 rounded px-2 py-1 outline-none hover:border-amber-500/50 transition-colors"
          >
            <option value="default">Global Default</option>
            {TTS_VOICES.map((v) => (
              <option key={v.id} value={v.id}>{v.label}</option>
            ))}
          </select>
        ) : eng === 'voxcpm' ? (
          <div className="flex items-center gap-2">
            <label className={`text-[10px] text-purple-400 border border-slate-700 rounded px-2 py-1 ${!sub.refAudioFile ? 'hover:border-purple-500 cursor-pointer' : ''} bg-slate-900 transition-colors flex items-center gap-1 min-h-[26px]`}>
               <Music className="w-3 h-3 shrink-0" />
               {sub.refAudioFile ? (
                 <div className="flex items-center gap-1.5 overflow-hidden">
                   <span className="truncate max-w-[50px] inline-block" title={sub.refAudioFile.name}>{sub.refAudioFile.name}</span>
                   <button 
                      className="hover:text-slate-200 text-slate-400 transition-colors ml-1 p-0.5 rounded hover:bg-slate-700" 
                      onClick={(e) => { 
                        e.preventDefault(); 
                        e.stopPropagation(); 
                        playAudioFile(sub.refAudioFile as File); 
                      }}
                      title="Play Subtitle Ref Audio"
                   >
                     <Play className="w-3 h-3 fill-current" />
                   </button>
                   <button 
                      className="hover:text-red-400 text-slate-400 transition-colors p-0.5 rounded hover:bg-red-500/20" 
                      onClick={(e) => { 
                        e.preventDefault(); 
                        e.stopPropagation(); 
                        updateSubtitles(prev => prev.map(s => s.id === sub.id ? { ...s, refAudioFile: undefined, refAudioBase64: undefined } : s)); 
                      }}
                      title="Remove Subtitle Ref Audio"
                   >
                     <X className="w-3 h-3" />
                   </button>
                 </div>
               ) : (
                 <span>Add Ref</span>
               )}
               {!sub.refAudioFile && <input type="file" accept="audio/*" onClick={(e) => e.stopPropagation()} onChange={(e) => handleSubReferenceAudioUpload(sub.id, e)} className="hidden" />}
            </label>
            <select
              value={sub.voice === 'default' ? '' : sub.voice}
              onClick={(e) => e.stopPropagation()}
              onChange={(e) => handleVoiceChange(sub.id, e.target.value || 'default')}
              className="w-32 bg-slate-900 text-[10px] text-purple-400/80 border border-slate-700 rounded px-2 py-1 outline-none hover:border-purple-500/50 transition-colors"
              title={(sub.refAudioFile || referenceAudioFile) ? "Ref Text (Auto if empty)" : `Prompt Text (Default: ${defaultVoxCPMVoice})`}
            >
              <option value="">Global Default</option>
              {VOXCPM_VOICES.map((v) => (
                <option key={v.id} value={v.id}>{v.label}</option>
              ))}
            </select>
          </div>
        ) : (
          <span className="text-[10px] text-slate-600 italic">Default Voice</span>
        );
        })()}
      </div>
    </div>
  );
});

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string>('');
  
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const onPlay = () => setIsPlaying(true);
    const onPause = () => setIsPlaying(false);

    video.addEventListener('play', onPlay);
    video.addEventListener('pause', onPause);

    return () => {
      video.removeEventListener('play', onPlay);
      video.removeEventListener('pause', onPause);
    };
  }, [videoUrl]);

  const [subtitles, setSubtitles] = useState<Subtitle[]>([]);
  const [isGeneratingAll, setIsGeneratingAll] = useState(false);
  const [draggingSub, setDraggingSub] = useState<{ id: number; offset: number; type: 'text' | 'audio' | 'trim-audio-start' | 'trim-audio-end' | 'trim-text-start' | 'trim-text-end' } | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [videoDuration, setVideoDuration] = useState(0);
  const [isScrubbing, setIsScrubbing] = useState(false);
  const isScrubbingRef = useRef(false);
  const [isPlaying, setIsPlaying] = useState(false);
  
  const [timelineHeight, setTimelineHeight] = useState(() => {
    const saved = localStorage.getItem('timeline_height');
    return saved ? parseInt(saved, 10) : 224;
  });
  const [zoomLevel, setZoomLevel] = useState(20); // px per second
  const [isResizingTimeline, setIsResizingTimeline] = useState(false);
  const [followPlayhead, setFollowPlayhead] = useState(true);
  const [showShortcuts, setShowShortcuts] = useState(false);

  const audioOverlaps = useMemo(() => {
    const overlaps = new Set<number>();
    if (subtitles.length > 1) {
      for (let i = 0; i < subtitles.length; i++) {
        const s1 = subtitles[i];
        if (!s1.audioUrl) continue;
        
        const start1 = s1.audioStartTime ?? s1.startTime;
        const dur1 = Math.max(0.01, (s1.audioTrimEnd ?? s1.audioDuration ?? (s1.endTime - s1.startTime)) - (s1.audioTrimStart ?? 0));
        const end1 = start1 + dur1;

        for (let j = i + 1; j < subtitles.length; j++) {
          const s2 = subtitles[j];
          if (!s2.audioUrl) continue;

          const start2 = s2.audioStartTime ?? s2.startTime;
          const dur2 = Math.max(0.01, (s2.audioTrimEnd ?? s2.audioDuration ?? (s2.endTime - s2.startTime)) - (s2.audioTrimStart ?? 0));
          const end2 = start2 + dur2;

          if (start1 < end2 - 0.01 && end1 > start2 + 0.01) {
            overlaps.add(s1.id);
            overlaps.add(s2.id);
          }
        }
      }
    }
    return overlaps;
  }, [subtitles]);

  const maxEndTime = useMemo(() => {
    if (subtitles.length === 0) return 0;
    // Faster max calculation
    let max = 0;
    for (let i = 0; i < subtitles.length; i++) {
      if (subtitles[i].endTime > max) max = subtitles[i].endTime;
    }
    return max;
  }, [subtitles]);

  const totalDuration = Math.max(videoDuration, maxEndTime) || 1;

  useEffect(() => {
    if (isResizingTimeline) {
      const handleMouseMove = (e: MouseEvent) => {
        const h = window.innerHeight - e.clientY;
        const boundedH = Math.max(100, Math.min(h, window.innerHeight * 0.8));
        setTimelineHeight(boundedH);
      };
      const handleMouseUp = () => {
        setIsResizingTimeline(false);
        localStorage.setItem('timeline_height', timelineHeight.toString());
        document.body.style.cursor = 'default';
      };
      document.body.style.cursor = 'row-resize';
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      return () => {
        window.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isResizingTimeline, timelineHeight]);

  const togglePlayback = useCallback(() => {
    if (!videoRef.current) return;
    if (videoRef.current.paused) {
      videoRef.current.play().catch(console.error);
    } else {
      videoRef.current.pause();
    }
  }, []);

  // Refs for keydown listener to avoid frequent re-subscribing
  const stateRef = useRef({ togglePlayback, subtitles, currentTime, videoRef });
  useEffect(() => {
    stateRef.current = { togglePlayback, subtitles, currentTime, videoRef };
  }, [togglePlayback, subtitles, currentTime, videoRef]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (document.activeElement instanceof HTMLInputElement || 
          document.activeElement instanceof HTMLTextAreaElement ||
          (document.activeElement as HTMLElement)?.isContentEditable) {
        return;
      }
      
      const { togglePlayback, subtitles, currentTime, videoRef } = stateRef.current;

      if (e.code === 'Space') {
        e.preventDefault();
        togglePlayback();
      } else if (e.code === 'ArrowLeft') {
        if (videoRef.current) {
          e.preventDefault();
          videoRef.current.currentTime = Math.max(0, videoRef.current.currentTime - (e.shiftKey ? 1 : 0.1));
        }
      } else if (e.code === 'ArrowRight') {
        if (videoRef.current) {
          e.preventDefault();
          videoRef.current.currentTime = Math.min(videoRef.current.duration || 0, videoRef.current.currentTime + (e.shiftKey ? 1 : 0.1));
        }
      } else if (e.code === 'ArrowUp') {
        e.preventDefault();
        // Jump to previous subtitle
        const prevSub = [...subtitles].reverse().find(s => s.startTime < currentTime - 0.1);
        if (prevSub && videoRef.current) {
          videoRef.current.currentTime = prevSub.startTime;
        }
      } else if (e.code === 'ArrowDown') {
        e.preventDefault();
        // Jump to next subtitle
        const nextSub = subtitles.find(s => s.startTime > currentTime + 0.1);
        if (nextSub && videoRef.current) {
          videoRef.current.currentTime = nextSub.startTime;
        }
      } else if (e.code === 'Delete' || e.code === 'Backspace') {
        // Only delete if something is active and we're not typing
        const activeSub = subtitles.find(s => currentTime >= s.startTime && currentTime <= s.endTime);
        if (activeSub) {
          e.preventDefault();
          updateSubtitles(prev => prev.filter(s => s.id !== activeSub.id));
        }
      } else if (e.code === 'Slash' && e.shiftKey) {
        e.preventDefault();
        setShowShortcuts(prev => !prev);
      } else if (e.code === 'Equal' || e.code === 'KeyV' || e.code === 'NumpadAdd') { // + key
        if (e.metaKey || e.ctrlKey) {
          e.preventDefault();
          setZoomLevel(prev => Math.min(500, prev * 1.2));
        }
      } else if (e.code === 'Minus' || e.code === 'NumpadSubtract') { // - key
        if (e.metaKey || e.ctrlKey) {
          e.preventDefault();
          setZoomLevel(prev => Math.max(2, prev / 1.2));
        }
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [togglePlayback]); // Only depend on stable togglePlayback

  const [history, setHistory] = useState<Subtitle[][]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);

  const pushToHistory = (newSubs: Subtitle[]) => {
    if (history.length > 0 && historyIndex >= 0) {
      const last = history[historyIndex];
      // Faster shallow check before stringify
      if (last.length === newSubs.length) {
        let changed = false;
        for (let i = 0; i < last.length; i++) {
          const s1 = last[i];
          const s2 = newSubs[i];
          if (s1.startTime !== s2.startTime || 
              s1.endTime !== s2.endTime || 
              s1.text !== s2.text || 
              s1.audioStartTime !== s2.audioStartTime ||
              s1.audioTrimStart !== s2.audioTrimStart ||
              s1.audioTrimEnd !== s2.audioTrimEnd ||
              s1.isLinked !== s2.isLinked ||
              s1.audioUrl !== s2.audioUrl) {
            changed = true;
            break;
          }
        }
        if (!changed) return;
      }
    }

    const snapshot = JSON.parse(JSON.stringify(newSubs));
    const newHistory = history.slice(0, historyIndex + 1);
    newHistory.push(snapshot);
    if (newHistory.length > 50) newHistory.shift();
    setHistory(newHistory);
    setHistoryIndex(newHistory.length - 1);
  };

  const undo = () => {
    if (historyIndex > 0) {
      const prevIndex = historyIndex - 1;
      const snapshot = JSON.parse(JSON.stringify(history[prevIndex]));
      setSubtitles(snapshot);
      setHistoryIndex(prevIndex);
    }
  };

  const redo = () => {
    if (historyIndex < history.length - 1) {
      const nextIndex = historyIndex + 1;
      const snapshot = JSON.parse(JSON.stringify(history[nextIndex]));
      setSubtitles(snapshot);
      setHistoryIndex(nextIndex);
    }
  };

  const updateSubtitles = (updater: Subtitle[] | ((prev: Subtitle[]) => Subtitle[]), skipHistory: boolean = false) => {
    setSubtitles(prev => {
      let next = typeof updater === 'function' ? updater(prev) : updater;
      
      // Ultimate Sanitization Guard
      next = next.map(s => {
        const startTime = Number.isFinite(s.startTime) ? Math.max(0, s.startTime) : 0;
        let endTime = Number.isFinite(s.endTime) ? s.endTime : startTime + 1;
        if (endTime <= startTime) endTime = startTime + 0.1;

        return {
          ...s,
          startTime,
          endTime,
          audioStartTime: (s.audioStartTime === undefined || !Number.isFinite(s.audioStartTime)) ? startTime : s.audioStartTime,
          audioTrimStart: Number.isFinite(s.audioTrimStart) ? Math.max(0, s.audioTrimStart) : 0,
          audioTrimEnd: (s.audioTrimEnd === undefined || !Number.isFinite(s.audioTrimEnd)) ? undefined : s.audioTrimEnd,
          audioDuration: (s.audioDuration === undefined || !Number.isFinite(s.audioDuration)) ? undefined : s.audioDuration,
        };
      });

      if (!skipHistory) {
        setTimeout(() => pushToHistory(next), 50);
      }
      return next;
    });
  };

  const timelineRulerCanvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = timelineRulerCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Handle high DPI screens
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, rect.width, rect.height);
    
    // Background background for the ruler area
    ctx.fillStyle = 'rgba(15, 23, 42, 0.5)'; // slate-900/50
    ctx.fillRect(0, 0, rect.width, 24);
    
    ctx.strokeStyle = 'rgba(100, 116, 139, 0.4)'; // slate-500/40
    ctx.fillStyle = 'rgba(203, 213, 225, 0.8)'; // slate-300
    ctx.font = '500 9px ui-monospace, monospace';
    ctx.textAlign = 'center';

    const totalD = totalDuration;
    if (!Number.isFinite(totalD) || totalD <= 0) return;
    
    // Adaptive steps based on zoomLevel (pixels per second)
    let majorStep = 5;
    if (zoomLevel > 150) majorStep = 0.5;
    else if (zoomLevel > 80) majorStep = 1;
    else if (zoomLevel > 40) majorStep = 2;
    else if (zoomLevel > 10) majorStep = 5;
    else if (zoomLevel > 5) majorStep = 10;
    else majorStep = 30;

    let minorStep = majorStep / 5;
    
    const maxMajorMarks = 500;
    if (totalD / majorStep > maxMajorMarks) {
      majorStep = Math.ceil(totalD / maxMajorMarks);
      minorStep = majorStep / 5;
    }

    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let s = 0; s <= totalD; s += majorStep) {
      const x = (s / totalD) * rect.width;
      
      // Major Tick
      ctx.moveTo(x, 12);
      ctx.lineTo(x, 24);
      
      const timeStr = s % 1 === 0 ? formatTime(s) : `${formatTime(Math.floor(s))}.${Math.floor((s % 1) * 10)}`;
      ctx.fillText(timeStr, x, 10);
      
      // Vertical Grid Line (very subtle)
      ctx.save();
      ctx.strokeStyle = 'rgba(51, 65, 85, 0.3)'; // slate-700/30
      ctx.beginPath();
      ctx.moveTo(x, 24);
      ctx.lineTo(x, rect.height + 1000); 
      ctx.stroke();
      ctx.restore();
      
      for (let m = minorStep; m < majorStep; m += minorStep) {
        if (s + m <= totalD) {
          const mx = ((s + m) / totalD) * rect.width;
          // Minor Tick
          ctx.moveTo(mx, 18);
          ctx.lineTo(mx, 24);
        }
      }
    }
    ctx.stroke();
  }, [totalDuration, videoDuration, zoomLevel, draggingSub === null]); // Redraw on size/duration/zoom changes

  const timelineRuler = <canvas ref={timelineRulerCanvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />;

  const handleTimelineScrub = (time: number) => {
    if (!videoRef.current) return;
    
    let currentTotal = Math.max(videoDuration, maxEndTime) || 1;
    if (!Number.isFinite(currentTotal)) currentTotal = 1;
    const clampedTime = Math.max(0, Math.min(time, currentTotal));
    
    if (Number.isFinite(clampedTime) && Math.abs(videoRef.current.currentTime - clampedTime) > 0.001) {
      videoRef.current.currentTime = clampedTime;
      setCurrentTime(clampedTime);
    }
    
    if (activeSubtitleIdRef.current !== null && currentAudioRef.current) {
        let sub = subtitles.find(s => s.id === activeSubtitleIdRef.current);
        if (sub && clampedTime >= sub.startTime && clampedTime <= sub.endTime) {
           const offset = sub.isLinked ? sub.startTime : (sub.audioStartTime ?? sub.startTime);
           const audioStartInFile = sub.audioTrimStart ?? 0;
           const audioDur = sub.audioDuration || (sub.endTime - sub.startTime);
           const trimEnd = sub.audioTrimEnd ?? audioDur;
           
           const expectedAudioTime = (clampedTime - offset) + audioStartInFile;
           const safeAudioTime = Math.max(audioStartInFile, Math.min(trimEnd, expectedAudioTime));
           
           if (Math.abs(currentAudioRef.current.currentTime - safeAudioTime) > 0.01) {
               currentAudioRef.current.currentTime = safeAudioTime;
           }
           
           if (expectedAudioTime > trimEnd + 0.02) {
             currentAudioRef.current.pause();
           }
           return;
        }
    }

    const newActiveSub = subtitles.find(s => {
      if (!s.audioUrl) return false;
      const start = s.isLinked ? s.startTime : (s.audioStartTime ?? s.startTime);
      const audioDur = s.audioDuration || (s.endTime - s.startTime);
      const effectiveDuration = (s.audioTrimEnd ?? audioDur) - (s.audioTrimStart ?? 0);
      return clampedTime >= start && clampedTime <= (start + effectiveDuration);
    });

    if (newActiveSub) {
      const audioStart = newActiveSub.isLinked ? newActiveSub.startTime : (newActiveSub.audioStartTime ?? newActiveSub.startTime);
      const audioTrimStart = newActiveSub.audioTrimStart ?? 0;
      const audioTime = (clampedTime - audioStart) + audioTrimStart;
      
      if (currentAudioRef.current) {
        currentAudioRef.current.pause();
      }
      
      const audio = new Audio(newActiveSub.audioUrl!);
      audio.currentTime = audioTime;
      audio.playbackRate = videoRef.current.playbackRate;
      currentAudioRef.current = audio;
      activeSubtitleIdRef.current = newActiveSub.id;
    } else {
      if (currentAudioRef.current) {
        currentAudioRef.current.pause();
        currentAudioRef.current = null;
      }
      activeSubtitleIdRef.current = null;
    }
  };

  const currentAudioRef = useRef<HTMLAudioElement | null>(null);
  const activeSubtitleIdRef = useRef<number | null>(null);
  const previewAudioRef = useRef<HTMLAudioElement | null>(null);
  const timelineContainerRef = useRef<HTMLDivElement>(null);
  const [previewingId, setPreviewingId] = useState<number | null>(null);

  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [ttsEngine, setTtsEngine] = useState<'gemini' | 'google-free' | 'voxcpm'>('google-free');

  const [referenceAudioFile, setReferenceAudioFile] = useState<File | null>(null);
  const [referenceAudioBase64, setReferenceAudioBase64] = useState<string | null>(null);

  const toggleSubtitleSelection = useCallback((id: number) => {
    setSelectedSubtitles(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const [selectedSubtitles, setSelectedSubtitles] = useState<Set<number>>(new Set());
  const [isBatchEditMode, setIsBatchEditMode] = useState(false);
  const [batchFindText, setBatchFindText] = useState('');
  const [batchReplaceText, setBatchReplaceText] = useState('');
  const [batchTimeShift, setBatchTimeShift] = useState<string>('0');

  const [playingAudio, setPlayingAudio] = useState<HTMLAudioElement | null>(null);

  const playAudioFile = (file: File) => {
    if (playingAudio) {
      playingAudio.pause();
    }
    const url = URL.createObjectURL(file);
    const audio = new Audio(url);
    audio.onended = () => URL.revokeObjectURL(url);
    audio.play().catch(console.error);
    setPlayingAudio(audio);
  };

  const handleSelectAll = () => {
    setSelectedSubtitles(new Set(subtitles.map(s => s.id)));
  };

  const handleDeselectAll = () => {
    setSelectedSubtitles(new Set());
  };

  const handleBatchReplace = () => {
    if (!batchFindText) return;
    updateSubtitles(prev => prev.map(s => {
      if (selectedSubtitles.has(s.id)) {
        return {
          ...s,
          text: s.text.split(batchFindText).join(batchReplaceText)
        };
      }
      return s;
    }));
  };

  const handleBatchTimeShift = () => {
    const shift = parseFloat(batchTimeShift);
    if (isNaN(shift)) return;
    updateSubtitles(prev => prev.map(s => {
      if (selectedSubtitles.has(s.id)) {
        const duration = s.endTime - s.startTime;
        const newStart = Math.max(0, s.startTime + shift);
        return {
          ...s,
          startTime: newStart,
          endTime: newStart + duration,
          audioStartTime: s.isLinked ? newStart : (s.audioStartTime ?? s.startTime) + shift
        };
      }
      return s;
    }));
  };

  const handleReferenceAudioUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setReferenceAudioFile(file);
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result as string;
        const base64 = result.split(',')[1];
        setReferenceAudioBase64(base64);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSubReferenceAudioUpload = (id: number, e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result as string;
        const base64 = result.split(',')[1];
        updateSubtitles(prev => prev.map(s => {
          if (s.id === id) {
            return {
              ...s,
              refAudioFile: file,
              refAudioBase64: base64
            };
          }
          return s;
        }));
      };
      reader.readAsDataURL(file);
    }
  };

  // Load video file
  const handleVideoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setVideoFile(file);
      setVideoUrl(URL.createObjectURL(file));
      setErrorMsg(null);
    }
  };

  // Load and parse SRT
  const handleSRTUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      const reader = new FileReader();
      reader.onload = (event) => {
        const text = event.target?.result as string;
        const parsed = parseSRT(text);
        updateSubtitles(parsed);
        setErrorMsg(null);
      };
      reader.readAsText(file);
    }
  };

  const [showSettings, setShowSettings] = useState(false);
  const [geminiKey, setGeminiKey] = useState(() => localStorage.getItem('gemini_api_key') || '');
  const [voxcpmUrl, setVoxcpmUrl] = useState(() => localStorage.getItem('voxcpm_url') || 'http://127.0.0.1:8808');

  const [defaultGeminiVoice, setDefaultGeminiVoice] = useState(() => localStorage.getItem('default_gemini_voice') || 'Puck');
  const [defaultVoxCPMVoice, setDefaultVoxCPMVoice] = useState(() => localStorage.getItem('default_voxcpm_voice') || 'alloy');

  const saveSettings = () => {
    localStorage.setItem('gemini_api_key', geminiKey);
    localStorage.setItem('voxcpm_url', voxcpmUrl);
    localStorage.setItem('default_gemini_voice', defaultGeminiVoice);
    localStorage.setItem('default_voxcpm_voice', defaultVoxCPMVoice);
    setShowSettings(false);
  };

  // Generate audio for a single subtitle
  const generateSubtitleAudio = async (sub: Subtitle): Promise<string | null> => {
    try {
      const engineToUse = (!sub.engine || sub.engine === 'default') ? ttsEngine : sub.engine;
      let voiceToUse = sub.voice;
      const isCloning = !!(sub.refAudioBase64 || referenceAudioBase64 || sub.refAudioFile || referenceAudioFile);

      if (!voiceToUse || voiceToUse === 'default') {
        if (engineToUse === 'gemini') voiceToUse = defaultGeminiVoice;
        else if (engineToUse === 'voxcpm') {
           voiceToUse = isCloning ? '' : defaultVoxCPMVoice;
        }
        else voiceToUse = 'km'; // fallback
      }

      if (engineToUse === 'voxcpm') {
        const baseURL = (localStorage.getItem('voxcpm_url') || 'http://127.0.0.1:8808')
          .replace(/\/$/, '');

        let refWavPayload = null;
        const finalRefBase64 = sub.refAudioBase64 || referenceAudioBase64;
        const finalRefFile = sub.refAudioFile || referenceAudioFile;
        
        if (finalRefFile) {
          const formData = new FormData();
          formData.append("files", finalRefFile);
          
          try {
            const uploadRes = await fetch(`${baseURL}/gradio_api/upload`, {
              method: 'POST',
              body: formData
            });
            
            if (!uploadRes.ok) {
              throw new Error(`Failed to upload reference audio: ${uploadRes.status}`);
            }
            
            const uploadedPaths = await uploadRes.json();
            if (Array.isArray(uploadedPaths) && uploadedPaths.length > 0) {
              refWavPayload = {
                path: uploadedPaths[0],
                orig_name: finalRefFile.name,
                meta: { _type: "gradio.FileData" }
              };
            }
          } catch (uploadErr) {
            console.warn("Gradio /upload failed, attempting base64 fallback...", uploadErr);
            // Fallback for older Gradio versions
            if (finalRefBase64) {
              const mimeType = finalRefFile.type || 'audio/wav';
              const dataUri = `data:${mimeType};base64,${finalRefBase64}`;
              refWavPayload = {
                data: dataUri,
                name: finalRefFile.name,
                orig_name: finalRefFile.name,
                meta: { _type: "gradio.FileData" }
              };
            }
          }
        }

        let promptText = voiceToUse && voiceToUse !== 'default' ? voiceToUse : '';
        if (refWavPayload && (promptText === defaultVoxCPMVoice || promptText.toLowerCase() === 'alloy')) {
            promptText = '';
        }

        const payload = {
          data: [
            sub.text,          // text
            "",                // control_instruction
            refWavPayload,     // reference_wav
            !!promptText,      // use_prompt_text
            promptText,        // prompt_text
            2.0,               // cfg_value
            false,             // normalize
            false,             // denoise
            10                 // dit_steps
          ]
        };

        let startRes;
        try {
          startRes = await fetch(`${baseURL}/gradio_api/call/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'ngrok-skip-browser-warning': 'true' },
            body: JSON.stringify(payload)
          });
        } catch (err: any) {
          if (err.name === 'TypeError' && err.message.includes('Failed to fetch')) {
            throw new Error(`Failed to connect to VoxCPM at ${baseURL}. If you are seeing a "Failed to fetch" or "Mixed Content" error, it's because this app is served over HTTPS but your VoxCPM server is HTTP. Please use ngrok (e.g. \`ngrok http 8808\`) to get an HTTPS URL for your local server, or allow insecure content in your browser settings.`);
          }
          throw err;
        }

        if (!startRes.ok) {
          throw new Error(`VoxCPM start error: ${startRes.status} - ${await startRes.text()}`);
        }

        const startData = await startRes.json();
        const eventId = startData.event_id;

        const resultRes = await fetch(
          `${baseURL}/gradio_api/call/generate/${eventId}`,
          {
            headers: {
              'ngrok-skip-browser-warning': 'true'
            }
          }
        );

        if (!resultRes.ok) {
          throw new Error(`VoxCPM result error: ${resultRes.status} - ${await resultRes.text()}`);
        }

        const resultText = await resultRes.text();

        const match = resultText.match(/data:\s*(\[.*\])/s);
        if (!match) {
          throw new Error(`VoxCPM returned no audio: ${resultText}`);
        }

        const parsed = JSON.parse(match[1]);
        const audioPath = parsed?.[0]?.url || parsed?.[0]?.path;

        if (!audioPath) {
          throw new Error(`VoxCPM audio URL not found: ${resultText}`);
        }

        const audioUrl = audioPath.startsWith('http')
          ? audioPath
          : `${baseURL}${audioPath}`;

        const audioRes = await fetch(audioUrl, {
          headers: {
            'ngrok-skip-browser-warning': 'true'
          }
        });
        const blob = await audioRes.blob();

        return URL.createObjectURL(blob);
      }

      if (engineToUse === 'google-free') {
        const res = await fetch('/api/tts', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: sub.text, lang: 'km' })
        });
        if (!res.ok) {
          const errData = await res.json();
          throw new Error(errData.error || 'Failed to fetch free TTS');
        }
        const data = await res.json();
        
        if (!data.results || data.results.length === 0) {
          throw new Error('No audio returned');
        }

        // Fix: googleTTS returns base64 MP3 chunks. We encode them to a single MP3 blob.
        const base64Chunks = data.results.map((r: any) => r.base64);
        const audioData = base64Chunks.join('');
        const binary = atob(audioData);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
          bytes[i] = binary.charCodeAt(i);
        }
        const blob = new Blob([bytes], { type: 'audio/mp3' });
        return URL.createObjectURL(blob);
      }

      // Create a prompt that encourages dramatic, expressive Khmer
      const prompt = `Read the following Khmer text vividly and passionately, with deeply emotional and dramatic tone resembling a Chinese short video drama: ${sub.text}`;
      
      const localGeminiKey = localStorage.getItem('gemini_api_key');
      const apiKeyToUse = (localGeminiKey && localGeminiKey.trim() !== '') ? localGeminiKey.trim() : process.env.GEMINI_API_KEY;
      
      if (!apiKeyToUse || apiKeyToUse === 'undefined' || apiKeyToUse === 'null') {
          throw new Error('API_KEY_INVALID: No Gemini API Key found.');
      }

      const activeAi = new GoogleGenAI({ apiKey: apiKeyToUse });
      
      const response = await activeAi.models.generateContent({
        model: 'gemini-3.1-flash-tts-preview',
        contents: [{ parts: [{ text: prompt }] }],
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: { voiceName: voiceToUse },
            },
          },
        },
      });

      const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
      if (!base64Audio) {
        throw new Error('No audio data received');
      }

      // Decode base64 PCM16 into WAV blob
      const binary = atob(base64Audio);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i);
      }
      const pcm16 = new Int16Array(bytes.buffer);
      const wavBlob = encodeWAV(pcm16, 24000); // TTS endpoint sample rate is 24kHz

      return URL.createObjectURL(wavBlob);
    } catch (err: any) {
      console.error('Failed to generate audio for subtitle', sub.id, err);
      let errMsg = err.message || String(err);
      let advice = "";

      if (errMsg.includes('429') || errMsg.includes('RESOURCE_EXHAUSTED')) {
         errMsg = "Rate limit or quota exceeded.";
         advice = " Please wait a few minutes before trying again or use a different API key.";
      } else if (errMsg.includes('403') || errMsg.includes('PERMISSION_DENIED') || errMsg.includes('API_KEY_INVALID')) {
         errMsg = "API Key error.";
         advice = " Please check your Gemini API key in the 'Settings' tab. Ensure it's active and has access to 'gemini-3.1-flash-tts-preview'.";
      } else if (errMsg.includes('401') || errMsg.includes('UNAUTHENTICATED')) {
         errMsg = "Unauthenticated request.";
         advice = " Ensure your API key is correct and properly configured in Settings.";
      } else if (errMsg.includes('NetworkError') || errMsg.includes('Failed to fetch')) {
         errMsg = "Network error.";
         advice = " Please check your internet connection and ensure the TTS service is accessible.";
      } else if (errMsg.includes('No audio data received') || errMsg.includes('No audio returned')) {
         errMsg = "Empty audio response.";
         advice = " The model was unable to generate speech for this text. Try modifying the text or using a different voice.";
      } else if (errMsg.includes('Safety') || errMsg.includes('SAFETY')) {
         errMsg = "Safety filter trigger.";
         advice = " The content was flagged by the safety filter. Try rephrasing the text.";
      }

      setErrorMsg(`${errMsg}${advice}`);
      return null;
    }
  };

  const handleVoiceChange = (id: number, voice: string) => {
    updateSubtitles((prev) => prev.map((s) => (s.id === id ? { ...s, voice, audioUrl: undefined, waveform: undefined } : s)));
  };

  const handleTimelineDrag = useCallback((subId: number, deltaX: number, type: 'text' | 'audio' | 'trim-audio-start' | 'trim-audio-end' | 'trim-text-start' | 'trim-text-end') => {
    if (!Number.isFinite(deltaX)) return;
    setDraggingSub({ id: subId, offset: deltaX, type });
  }, []);

  const handleTimelineDragEnd = useCallback((subId: number, info: any, type: 'text' | 'audio' | 'trim-audio-start' | 'trim-audio-end' | 'trim-text-start' | 'trim-text-end') => {
    const deltaX = info.offset.x || 0;
    setDraggingSub(null);
    const deltaTime = zoomLevel > 0 ? deltaX / zoomLevel : 0;
    
    if (isNaN(deltaTime) || Math.abs(deltaTime) < 0.001 && !type.startsWith('trim')) return;

    updateSubtitles(prev => prev.map(s => {
      if (s.id === subId) {
        const duration = s.endTime - s.startTime;
        const isLinked = s.isLinked ?? true;
        
        if (type === 'text') {
          const duration = Math.max(0.1, s.endTime - s.startTime);
          const newStart = Math.max(0, s.startTime + deltaTime);
          if (!Number.isFinite(newStart)) return s;
          return {
            ...s,
            startTime: newStart,
            endTime: newStart + duration,
            audioStartTime: isLinked ? newStart : (s.audioStartTime ?? s.startTime)
          };
        } else if (type === 'audio') {
          const currentAudioStart = s.audioStartTime ?? s.startTime;
          const newAudioStart = Math.max(0, currentAudioStart + deltaTime);
          if (!Number.isFinite(newAudioStart)) return s;
          const currentTextStart = s.startTime;
          const audioDur = Math.max(0.1, (s.audioTrimEnd ?? s.audioDuration ?? (s.endTime - s.startTime)) - (s.audioTrimStart ?? 0));
          return {
            ...s,
            startTime: isLinked ? newAudioStart : currentTextStart,
            endTime: isLinked ? newAudioStart + audioDur : s.endTime,
            audioStartTime: newAudioStart
          };
        } else if (type === 'trim-audio-start') {
          const currentTrimStart = s.audioTrimStart ?? 0;
          const audioDuration = s.audioDuration || (s.endTime - s.startTime);
          const currentTrimEnd = s.audioTrimEnd ?? audioDuration;
          const currentAudioStart = s.audioStartTime ?? s.startTime;
          
          const newTrimStart = Math.max(0, Math.min(currentTrimEnd - 0.1, currentTrimStart + deltaTime));
          if (!Number.isFinite(newTrimStart)) return s;
          const actualDelta = newTrimStart - currentTrimStart;
          const newAudioStart = currentAudioStart + actualDelta;
          const isLinked = s.isLinked ?? true;

          return {
            ...s,
            audioTrimStart: newTrimStart,
            audioStartTime: newAudioStart,
            startTime: isLinked ? newAudioStart : s.startTime,
            endTime: isLinked ? (newAudioStart + Math.max(0.1, currentTrimEnd - newTrimStart)) : s.endTime
          };
        } else if (type === 'trim-audio-end') {
          const audioDuration = s.audioDuration || Math.max(0.1, s.endTime - s.startTime);
          const currentTrimStart = s.audioTrimStart ?? 0;
          const currentTrimEnd = s.audioTrimEnd ?? audioDuration;
          const newTrimEnd = Math.max(currentTrimStart + 0.1, Math.min(audioDuration, currentTrimEnd + deltaTime));
          if (!Number.isFinite(newTrimEnd)) return s;
          const isLinked = s.isLinked ?? true;
          const currentAudioStart = s.audioStartTime ?? s.startTime;

          return {
            ...s,
            audioTrimEnd: newTrimEnd,
            endTime: isLinked ? (currentAudioStart + Math.max(0.1, newTrimEnd - currentTrimStart)) : s.endTime
          };
        } else if (type === 'trim-text-start') {
          const newStart = Math.max(0, Math.min(s.endTime - 0.1, s.startTime + deltaTime));
          if (!Number.isFinite(newStart)) return s;
          const isLinked = s.isLinked ?? true;
          return {
            ...s,
            startTime: newStart,
            audioStartTime: isLinked ? newStart : s.audioStartTime
          };
        } else if (type === 'trim-text-end') {
          const newEnd = Math.max(s.startTime + 0.1, s.endTime + deltaTime);
          if (!Number.isFinite(newEnd)) return s;
          return {
            ...s,
            endTime: newEnd,
          };
        }
      }
      return s;
    }), false);
  }, [zoomLevel, updateSubtitles]);

  const handleToggleLink = (id: number) => {
    updateSubtitles(prev => prev.map(s => {
      if (s.id === id) {
        const isLinked = !(s.isLinked ?? true);
        return {
          ...s,
          isLinked,
          audioStartTime: isLinked ? s.startTime : (s.audioStartTime ?? s.startTime)
        };
      }
      return s;
    }));
  };

  const handleEngineChange = (id: number, engine: string) => {
    updateSubtitles((prev) => prev.map((s) => (s.id === id ? { ...s, engine, voice: 'default', audioUrl: undefined, waveform: undefined } : s)));
  };

  const handlePreviewAudio = (e: React.MouseEvent, sub: Subtitle) => {
    e.stopPropagation();
    if (!sub.audioUrl) return;

    if (previewAudioRef.current) {
      previewAudioRef.current.pause();
      previewAudioRef.current.onended = null;
    }

    if (previewingId === sub.id) {
      setPreviewingId(null);
      return;
    }

    const audio = new Audio(sub.audioUrl);
    previewAudioRef.current = audio;
    setPreviewingId(sub.id);

    audio.onended = () => {
      setPreviewingId(null);
    };
    
    audio.play().catch(err => {
      console.error('Failed to play preview', err);
      setPreviewingId(null);
    });
  };

  // Generate audio for a single subtitle from UI
  const handleGenerateSingle = async (e: React.MouseEvent, sub: Subtitle) => {
    e.stopPropagation();
    setErrorMsg(null);
    setSubtitles((prev) => prev.map((s) => (s.id === sub.id ? { ...s, isGenerating: true } : s)));
    const url = await generateSubtitleAudio(sub);
    let duration = 0;
    let waveform: number[] | undefined;
    if (url) {
      duration = await getAudioDuration(url);
      waveform = await generateWaveform(url);
    }
    updateSubtitles((prev) =>
      prev.map((s) => (s.id === sub.id ? { ...s, isGenerating: false, audioUrl: url || undefined, audioDuration: duration || undefined, waveform } : s))
    );
    if (url) {
       // Stop any existing preview
       if (previewAudioRef.current) {
         previewAudioRef.current.pause();
       }
       const audio = new Audio(url);
       previewAudioRef.current = audio;
       setPreviewingId(sub.id);
       audio.onended = () => setPreviewingId(null);
       audio.play().catch(e => {
         console.error('Auto-play failed', e);
         setPreviewingId(null);
       });
    }
  };

  // Generate all sequentially
  const handleGenerateAll = async () => {
    setErrorMsg(null);
    setIsGeneratingAll(true);
    for (let i = 0; i < subtitles.length; i++) {
      if (subtitles[i].audioUrl) continue; // skip already generated

      // Optimistic update for loading state
      setSubtitles((prev) =>
        prev.map((s, idx) => (idx === i ? { ...s, isGenerating: true } : s))
      );

      const url = await generateSubtitleAudio(subtitles[i]);
      let duration = 0;
      let waveform: number[] | undefined;
      if (url) {
        duration = await getAudioDuration(url);
        waveform = await generateWaveform(url);
      }
      
      // Update with result
      updateSubtitles((prev) =>
        prev.map((s, idx) =>
          idx === i ? { ...s, isGenerating: false, audioUrl: url || undefined, audioDuration: duration || undefined, waveform } : s
        )
      );

      if (!url) {
        // Break the loop if there was an error (e.g., 429 quota exceeded)
        break;
      }

      // Auto-play the audio snippet we just generated, and wait for it to finish!
      if (previewAudioRef.current) previewAudioRef.current.pause();
      const audio = new Audio(url);
      previewAudioRef.current = audio;
      setPreviewingId(subtitles[i].id);
        
      await new Promise<void>((resolve) => {
         audio.onended = () => {
            setPreviewingId(null);
            resolve();
         };
         audio.play().catch((e) => {
            console.error('Auto-play failed', e);
            setPreviewingId(null);
            resolve();
         });
      });

      // Add a delay to avoid rate limiting (Gemini Free Tier is 15 RPM, so 4.1s per request to be safe)
      if (i < subtitles.length - 1) {
        await new Promise((res) => setTimeout(res, 4100));
      }
    }
    setIsGeneratingAll(false);
  };

  const [isExportingAudio, setIsExportingAudio] = useState(false);

  const handleExportAudioTrack = async () => {
    if (subtitles.length === 0) return;
    const totalDuration = Math.max(...subtitles.map(s => s.endTime));
    if (totalDuration <= 0) return;

    setIsExportingAudio(true);
    setErrorMsg(null);

    try {
      const OfflineCtxClass = window.OfflineAudioContext || (window as any).webkitOfflineAudioContext;
      const sampleRate = 44100;
      const offlineCtx = new OfflineCtxClass(1, Math.ceil(sampleRate * totalDuration), sampleRate);
      
      // Load all valid audios
      for (const sub of subtitles) {
        if (!sub.audioUrl) continue;
        try {
          const resp = await fetch(sub.audioUrl);
          const arrayBuffer = await resp.arrayBuffer();
          const audioBuffer = await offlineCtx.decodeAudioData(arrayBuffer);
          const source = offlineCtx.createBufferSource();
          source.buffer = audioBuffer;
          source.connect(offlineCtx.destination);
          const startAt = sub.isLinked ? sub.startTime : (sub.audioStartTime ?? sub.startTime);
          source.start(startAt);
        } catch (e) {
          console.warn(`Failed to process audio for subtitle ${sub.id}`, e);
        }
      }

      const renderedBuffer = await offlineCtx.startRendering();
      const float32Array = renderedBuffer.getChannelData(0);
      const int16Array = new Int16Array(float32Array.length);
      for (let i = 0; i < float32Array.length; i++) {
        let s = Math.max(-1, Math.min(1, float32Array[i]));
        int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }

      const wavBlob = encodeWAV(int16Array, renderedBuffer.sampleRate);
      const url = URL.createObjectURL(wavBlob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'synthesized_track.wav';
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Failed to export audio track:', err);
      setErrorMsg("Failed to export audio track.");
    } finally {
      setIsExportingAudio(false);
    }
  };

  // Synchronize audio playback with video
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    let syncFrame: number;

    const loop = () => {
      if (!videoRef.current) return;
      const video = videoRef.current;
      const time = video.currentTime;
      
      // Update state selectively to reduce re-renders. 
      // Throttle to ~33fps (every 30ms) to maintain smooth UI while reducing render load
      const now = performance.now();
      if (Number.isFinite(time) && (now - (window as any)._lastVideoStateUpdate > 30)) {
        setCurrentTime(time);
        (window as any)._lastVideoStateUpdate = now;
      }

      // Cache subtitles length
      const subsCount = subtitles.length;
      if (subsCount === 0) {
        syncFrame = requestAnimationFrame(loop);
        return;
      }
      
      // Find active audio (for playback) - more efficient loop
      let activeAudio: Subtitle | undefined;
      for (let i = 0; i < subsCount; i++) {
        const s = subtitles[i];
        if (!s.audioUrl) continue;
        const start = s.isLinked ? s.startTime : (s.audioStartTime ?? s.startTime);
        const audioDur = s.audioDuration || (s.endTime - s.startTime);
        const effectiveDuration = (s.audioTrimEnd ?? audioDur) - (s.audioTrimStart ?? 0);
        if (time >= start && time <= (start + effectiveDuration)) {
          activeAudio = s;
          break;
        }
      }

      if (activeAudio) {
        if (activeSubtitleIdRef.current !== activeAudio.id) {
          // Changed audio boundary
          if (currentAudioRef.current) {
            currentAudioRef.current.pause();
          }

          if (activeAudio.audioUrl) {
            const audio = new Audio(activeAudio.audioUrl);
            const start = activeAudio.isLinked ? activeAudio.startTime : (activeAudio.audioStartTime ?? activeAudio.startTime);
            const audioStartInFile = activeAudio.audioTrimStart ?? 0;
            audio.currentTime = (time - start) + audioStartInFile; // sync offset with trim
            // Match playback rate
            audio.playbackRate = video.playbackRate;
            if (!video.paused) {
              audio.play().catch(console.error);
            }
            currentAudioRef.current = audio;
          } else {
            currentAudioRef.current = null;
          }
          activeSubtitleIdRef.current = activeAudio.id;
        } else {
          // Currently in the same audio window
          if (currentAudioRef.current) {
            // Keep playback state synced
            if (video.paused && !currentAudioRef.current.paused) {
              currentAudioRef.current.pause();
            } else if (!video.paused && currentAudioRef.current.paused) {
              currentAudioRef.current.play().catch(console.error);
            }
            
            // Keep playback rate synced
            if (currentAudioRef.current.playbackRate !== video.playbackRate) {
              currentAudioRef.current.playbackRate = video.playbackRate;
            }

            // Sophisticated drift correction logic
            const start = activeAudio.isLinked ? activeAudio.startTime : (activeAudio.audioStartTime ?? activeAudio.startTime);
            const audioStartInFile = activeAudio.audioTrimStart ?? 0;
            const audioFileName = activeAudio.audioDuration || (activeAudio.endTime - activeAudio.startTime);
            const trimEnd = activeAudio.audioTrimEnd ?? audioFileName;
            
            // Recalculate absolute expected position
            const expectedTimeInFile = (time - start) + audioStartInFile;
            
            // Adaptive threshold based on interaction state
            // During scrubbing or seeking, we want sub-frame precision (1ms)
            // During normal playback, we allow a small buffer (80ms) to prevent audible "studdering" 
            // from constant currentTime seeks which can click.
            const isSeekingOrScrubbing = isScrubbingRef.current || video.seeking;
            const sensitivity = isSeekingOrScrubbing ? 0.005 : 0.08;
            
            // Also tighten threshold if speed is modified
            const adjustedSensitivity = Math.abs(video.playbackRate - 1) > 0.1 ? Math.min(sensitivity, 0.04) : sensitivity;
            
            const actualTime = currentAudioRef.current.currentTime;
            const drift = Math.abs(actualTime - expectedTimeInFile);
            
            if (drift > adjustedSensitivity) {
              // Clamp to valid audio range
              const safeTime = Math.max(audioStartInFile, Math.min(trimEnd, expectedTimeInFile));
              currentAudioRef.current.currentTime = safeTime;
              
              // Force pause if we've drifted significantly past the logical end of the clip
              if (expectedTimeInFile > trimEnd + 0.05) {
                if (!currentAudioRef.current.paused) currentAudioRef.current.pause();
              }
            }
          }
        }
      } else {
        // Outside of any audio
        if (activeSubtitleIdRef.current !== null) {
          if (currentAudioRef.current) {
            currentAudioRef.current.pause();
            currentAudioRef.current = null;
          }
          activeSubtitleIdRef.current = null;
        }
      }

      if (timelineContainerRef.current && !video.paused && !isScrubbingRef.current && followPlayhead) {
        const container = timelineContainerRef.current;
        // playheadX is pixels per second
        const playheadX = time * zoomLevel;
        const containerWidth = container.offsetWidth;
        const scrollLeft = container.scrollLeft;
        
        // Follow playhead with a smooth window: keep playhead between 20% and 80% of view
        const padding = containerWidth * 0.2;
        if (playheadX > scrollLeft + containerWidth - padding) {
          container.scrollLeft = playheadX - (containerWidth * 0.2);
        } else if (playheadX < scrollLeft + padding) {
          container.scrollLeft = Math.max(0, playheadX - (containerWidth * 0.8));
        }
      }

      syncFrame = requestAnimationFrame(loop);
    };

    const onSeeked = () => {
      // Force resync after seeking
      if (currentAudioRef.current && activeSubtitleIdRef.current !== null) {
        const sub = subtitles.find((s) => s.id === activeSubtitleIdRef.current);
        if (sub) {
          const start = sub.isLinked ? sub.startTime : (sub.audioStartTime ?? sub.startTime);
          currentAudioRef.current.currentTime = (video.currentTime - start) + (sub.audioTrimStart ?? 0);
        }
      }
    };

    video.addEventListener('seeked', onSeeked);
    syncFrame = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(syncFrame);
      video.removeEventListener('seeked', onSeeked);
      if (currentAudioRef.current) {
         currentAudioRef.current.pause();
         currentAudioRef.current = null;
      }
    };
  }, [subtitles]);

  // Clean up object URLs
  useEffect(() => {
    return () => {
      if (videoUrl) URL.revokeObjectURL(videoUrl);
      subtitles.forEach((s) => {
        if (s.audioUrl) URL.revokeObjectURL(s.audioUrl);
      });
    };
  }, []);

  return (
    <div className="w-full h-screen bg-[#020617] text-slate-200 font-sans overflow-hidden flex flex-col">
      <datalist id="voxcpm-voices">
        {VOXCPM_VOICES.map(v => <option key={v.id} value={v.id}>{v.label}</option>)}
      </datalist>

      {/* Top Header */}
      <header className="h-14 shrink-0 border-b border-slate-800 flex items-center justify-between px-6 bg-slate-900/50 backdrop-blur-md">
        {/* Shortcuts Modal */}
        {showShortcuts && (
          <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm" onClick={() => setShowShortcuts(false)}>
            <motion.div 
              initial={{ opacity: 0, scale: 0.9, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              className="bg-slate-900 border border-slate-800 rounded-xl shadow-2xl w-full max-w-lg overflow-hidden" 
              onClick={e => e.stopPropagation()}
            >
              <div className="p-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/50">
                <div className="flex items-center gap-2">
                  <HelpCircle className="w-5 h-5 text-amber-500" />
                  <h3 className="text-lg font-bold text-white">Keyboard Shortcuts</h3>
                </div>
                <button onClick={() => setShowShortcuts(false)} className="p-1 hover:bg-slate-800 rounded-full transition-colors text-slate-400">
                  <X className="w-5 h-5" />
                </button>
              </div>
              <div className="p-6 grid grid-cols-2 gap-6 text-sm">
                <div className="space-y-4">
                  <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest">Playback</h4>
                  <div className="flex justify-between items-center bg-slate-950/50 p-2 rounded border border-slate-800/50">
                    <span className="text-slate-400">Play / Pause</span>
                    <kbd className="px-2 py-1 bg-slate-800 rounded text-amber-500 font-mono text-xs shadow-sm shadow-black/40 border border-slate-700/50">Space</kbd>
                  </div>
                  <div className="flex justify-between items-center bg-slate-950/50 p-2 rounded border border-slate-800/50">
                    <span className="text-slate-400">Step Back</span>
                    <kbd className="px-2 py-1 bg-slate-800 rounded text-amber-500 font-mono text-xs shadow-sm shadow-black/40 border border-slate-700/50">←</kbd>
                  </div>
                  <div className="flex justify-between items-center bg-slate-950/50 p-2 rounded border border-slate-800/50">
                    <span className="text-slate-400">Step Forward</span>
                    <kbd className="px-2 py-1 bg-slate-800 rounded text-amber-500 font-mono text-xs shadow-sm shadow-black/40 border border-slate-700/50">→</kbd>
                  </div>
                  <div className="flex justify-between items-center bg-slate-950/50 p-2 rounded border border-slate-800/50">
                    <span className="text-slate-400">1s Jump</span>
                    <kbd className="px-2 py-1 bg-slate-800 rounded text-amber-500 font-mono text-xs shadow-sm shadow-black/40 border border-slate-700/50">Shift + ←/→</kbd>
                  </div>
                </div>
                <div className="space-y-4">
                  <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest">Editor</h4>
                  <div className="flex justify-between items-center bg-slate-950/50 p-2 rounded border border-slate-800/50">
                    <span className="text-slate-400">Prev Subtitle</span>
                    <kbd className="px-2 py-1 bg-slate-800 rounded text-amber-500 font-mono text-xs shadow-sm shadow-black/40 border border-slate-700/50">↑</kbd>
                  </div>
                  <div className="flex justify-between items-center bg-slate-950/50 p-2 rounded border border-slate-800/50">
                    <span className="text-slate-400">Next Subtitle</span>
                    <kbd className="px-2 py-1 bg-slate-800 rounded text-amber-500 font-mono text-xs shadow-sm shadow-black/40 border border-slate-700/50">↓</kbd>
                  </div>
                  <div className="flex justify-between items-center bg-slate-950/50 p-2 rounded border border-slate-800/50">
                    <span className="text-slate-400">Delete Clip</span>
                    <kbd className="px-2 py-1 bg-slate-800 rounded text-amber-500 font-mono text-xs shadow-sm shadow-black/40 border border-slate-700/50">Del / BS</kbd>
                  </div>
                  <div className="flex justify-between items-center bg-slate-950/50 p-2 rounded border border-slate-800/50">
                    <span className="text-slate-400">Keyboard help</span>
                    <kbd className="px-2 py-1 bg-slate-800 rounded text-amber-500 font-mono text-xs shadow-sm shadow-black/40 border border-slate-700/50">?</kbd>
                  </div>
                </div>
                <div className="col-span-2 space-y-4 pt-2">
                   <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest">Timeline</h4>
                   <div className="flex justify-between items-center bg-slate-950/50 p-2 rounded border border-slate-800/50">
                    <span className="text-slate-400">Zoom Canvas</span>
                    <kbd className="px-2 py-1 bg-slate-800 rounded text-amber-500 font-mono text-xs shadow-sm shadow-black/40 border border-slate-700/50">Ctrl / ⌘ + (+/-)</kbd>
                  </div>
                </div>
              </div>
              <div className="p-4 bg-slate-950/50 border-t border-slate-800 text-xs text-slate-500 text-center italic">
                You can also drag handles on the timeline for fine-tuning audio start/end.
              </div>
            </motion.div>
          </div>
        )}

        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-amber-500 rounded flex items-center justify-center text-slate-950 font-bold">
            K
          </div>
          <span className="font-semibold tracking-tight text-lg">KhmerDub <span className="text-amber-500">Studio</span></span>
        </div>
        <div className="flex items-center gap-4">
          <button onClick={() => setShowSettings(true)} className="p-2 hover:bg-slate-800 rounded-md text-slate-400 hover:text-white transition-colors">
            <Settings className="w-5 h-5" />
          </button>
          
          <div className="flex items-center gap-1 bg-slate-800/50 rounded-md p-1 mr-2">
            <button 
              onClick={undo} 
              disabled={historyIndex <= 0}
              className="p-1.5 hover:bg-slate-700 rounded text-slate-400 hover:text-slate-200 disabled:opacity-30 disabled:hover:bg-transparent"
              title="Undo"
            >
              <Undo2 className="w-4 h-4" />
            </button>
            <button 
              onClick={redo} 
              disabled={historyIndex >= history.length - 1}
              className="p-1.5 hover:bg-slate-700 rounded text-slate-400 hover:text-slate-200 disabled:opacity-30 disabled:hover:bg-transparent"
              title="Redo"
            >
              <Redo2 className="w-4 h-4" />
            </button>
          </div>

          <button className="px-4 py-1.5 bg-slate-800 rounded-md text-sm font-medium border border-slate-700 hidden md:block">រក្សាទុក</button>
          <button 
            disabled={isExportingAudio || subtitles.length === 0}
            onClick={handleExportAudioTrack}
            className="px-4 py-1.5 bg-amber-500 text-slate-950 rounded-md text-sm font-bold shadow-lg shadow-amber-500/20 disabled:opacity-50"
          >
            {isExportingAudio ? 'កំពុងនាំចេញ...' : 'នាំចេញសំឡេង'}
          </button>
          <button className="px-4 py-1.5 bg-amber-500/10 text-amber-500 border border-amber-500/20 rounded-md text-sm font-bold shadow-lg shadow-amber-500/5 hidden md:block">នាំចេញវីដេអូ</button>
        </div>
      </header>

      {/* Main Workspace */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left Panel: Upload Controls */}
        <aside className="w-72 border-r border-slate-800 p-5 flex flex-col gap-6 bg-[#020617] overflow-y-auto hidden md:flex shrink-0">
          <div>
            <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4">ឯកសារ (Files)</h3>
            <div className="space-y-4">
              {/* Video Upload */}
              <div className="relative border border-dashed border-slate-700 hover:border-amber-500 hover:bg-slate-800/50 transition-colors rounded p-4 flex flex-col items-center justify-center group cursor-pointer h-24">
                <Film className="w-5 h-5 text-slate-500 mb-2 group-hover:text-amber-400" />
                <span className="text-xs font-medium text-slate-400 group-hover:text-slate-200 text-center">
                  {videoFile ? videoFile.name : 'Upload MP4/WebM'}
                </span>
                <input 
                  type="file" 
                  accept="video/mp4,video/webm" 
                  onChange={handleVideoUpload}
                  className="absolute inset-0 opacity-0 cursor-pointer"
                />
              </div>

              {/* SRT Upload */}
              <div className="relative border border-dashed border-slate-700 hover:border-amber-500 hover:bg-slate-800/50 transition-colors rounded p-4 flex flex-col items-center justify-center group cursor-pointer h-24">
                <FileText className="w-5 h-5 text-slate-500 mb-2 group-hover:text-amber-400" />
                <span className="text-xs font-medium text-slate-400 group-hover:text-slate-200 text-center">
                  {subtitles.length > 0 ? `${subtitles.length} lines loaded` : 'Upload Khmer SRT'}
                </span>
                <input 
                  type="file" 
                  accept=".srt" 
                  onChange={handleSRTUpload}
                  className="absolute inset-0 opacity-0 cursor-pointer"
                />
              </div>

              {/* Global Reference Audio Upload */}
              <div className="relative border border-dashed border-slate-700 hover:border-purple-500 hover:bg-slate-800/50 transition-colors rounded p-4 flex flex-col items-center justify-center group cursor-pointer h-24">
                {!referenceAudioFile ? (
                  <>
                    <Music className="w-5 h-5 text-slate-500 mb-2 group-hover:text-purple-400" />
                    <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1">VoxCPM Clone Ref</span>
                    <span className="text-xs font-medium text-slate-400 group-hover:text-slate-200 text-center truncate max-w-[90%]">
                      Upload Global Ref Audio
                    </span>
                    <input 
                      type="file" 
                      accept="audio/*" 
                      onChange={handleReferenceAudioUpload}
                      className="absolute inset-0 opacity-0 cursor-pointer"
                    />
                  </>
                ) : (
                  <div className="flex w-full items-center justify-between z-10 relative">
                    <div className="flex items-center gap-2 overflow-hidden flex-1">
                       <Music className="w-5 h-5 text-purple-400 shrink-0" />
                       <div className="flex flex-col overflow-hidden text-left">
                         <span className="text-[10px] text-purple-400 font-bold uppercase truncate">Ref Selected</span>
                         <span className="text-[11px] text-slate-300 truncate" title={referenceAudioFile.name}>{referenceAudioFile.name}</span>
                       </div>
                    </div>
                    <div className="flex gap-1 shrink-0 ml-1">
                      <button 
                         onClick={(e) => { e.preventDefault(); e.stopPropagation(); playAudioFile(referenceAudioFile); }} 
                         className="p-1.5 bg-slate-800 border border-slate-700 hover:border-slate-500 hover:bg-slate-700 rounded text-slate-400 hover:text-slate-200 transition-colors"
                         title="Play Reference Audio"
                      >
                         <Play className="w-3.5 h-3.5 fill-current" />
                      </button>
                      <button 
                         onClick={(e) => { e.preventDefault(); e.stopPropagation(); setReferenceAudioFile(null); setReferenceAudioBase64(null); }} 
                         className="p-1.5 bg-slate-800 border border-slate-700 hover:border-red-500/50 hover:bg-red-500/20 rounded text-slate-400 hover:text-red-400 transition-colors"
                         title="Remove Reference Audio"
                      >
                         <X className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
          
          <div className="mt-auto p-4 bg-slate-900/80 border border-slate-800 rounded-xl space-y-4 text-center">
              {subtitles.length > 0 && (
                <button
                  onClick={handleGenerateAll}
                  disabled={isGeneratingAll}
                  className="w-full bg-amber-500 hover:bg-amber-400 text-slate-950 font-bold py-2 px-4 rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 text-[13px] shadow-lg shadow-amber-500/20"
                >
                  {isGeneratingAll ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Generating...
                    </>
                  ) : (
                    'Generate Audio'
                  )}
                </button>
              )}
          </div>
        </aside>

        {/* Center: Video Player Area */}
        <main className="flex-1 bg-black/40 flex flex-col relative overflow-hidden">
          <div className="flex-1 flex flex-col items-center justify-center p-4 md:p-8 relative overflow-y-auto">
            <div className="w-full max-w-4xl aspect-video bg-slate-900 rounded-lg shadow-2xl relative overflow-hidden ring-1 ring-slate-800 flex items-center justify-center shrink-0">
              {videoUrl ? (
                <video 
                  ref={videoRef}
                  src={videoUrl} 
                  controls 
                  crossOrigin="anonymous"
                  onLoadedMetadata={(e) => setVideoDuration(e.currentTarget.duration || 0)}
                  className="w-full h-full bg-black"
                />
              ) : (
                <div className="absolute inset-0 flex items-center justify-center flex-col gap-4 bg-gradient-to-t from-slate-950/60 to-transparent text-slate-600">
                  <Play className="w-12 h-12 opacity-30" />
                  <span className="text-sm font-medium">Waiting for video...</span>
                </div>
              )}
            </div>
            
            {/* Sync Warning */}
            {(() => {
              const activeSub = subtitles.find(s => currentTime >= s.startTime && currentTime <= s.endTime);
              if (activeSub && activeSub.audioUrl && activeSub.audioDuration) {
                 const subDuration = activeSub.endTime - activeSub.startTime;
                 if (activeSub.audioDuration > subDuration + 0.2) {
                    return (
                      <div className="w-full max-w-4xl mt-3 shrink-0 flex justify-center">
                        <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs rounded-full shadow-[0_0_10px_rgba(244,63,94,0.1)] transition-opacity">
                           <span className="relative flex h-2 w-2 items-center justify-center">
                             <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-rose-400 opacity-75"></span>
                             <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-rose-500"></span>
                           </span>
                           Audio drift detected: Synthesized audio ({activeSub.audioDuration.toFixed(1)}s) is longer than subtitle clip ({subDuration.toFixed(1)}s).
                           <button 
                             onClick={() => {
                               updateSubtitles(prev => prev.map(s => 
                                 s.id === activeSub.id ? { ...s, endTime: s.startTime + (s.audioDuration || 0) } : s
                               ));
                             }}
                             className="ml-2 px-2 py-0.5 bg-rose-500 text-white rounded hover:bg-rose-600 transition-colors font-bold whitespace-nowrap"
                           >
                             Extend to Fit
                           </button>
                        </div>
                      </div>
                    );
                 }
              }
              return null;
            })()}
          </div>
          
          {/* Audio Timeline Panel */}
          <div 
            className="shrink-0 border-t border-slate-800 bg-slate-950 flex flex-col overflow-hidden shadow-[inset_0_4px_20px_rgba(0,0,0,0.5)] relative"
            style={{ height: `${timelineHeight}px` }}
          >
            {/* Resize Handle */}
            <div 
              className="absolute top-0 left-0 right-0 h-1 cursor-row-resize z-50 hover:bg-amber-500/50 transition-colors"
              onMouseDown={() => setIsResizingTimeline(true)}
            />
            
            <div className="p-4 flex flex-col gap-2 h-full">
              <h3 className="text-xs font-medium text-slate-400 shrink-0 select-none flex justify-between items-center">
                 <div className="flex items-center gap-3">
                   <span>Timeline</span>
                   <button 
                    onClick={togglePlayback}
                    className="p-1 rounded-full bg-slate-800 border border-slate-700 hover:bg-slate-700 text-slate-300 hover:text-amber-400 transition-all active:scale-95 flex items-center justify-center"
                   >
                     {isPlaying ? <Pause className="w-3.5 h-3.5 fill-current" /> : <Play className="w-3.5 h-3.5 fill-current" />}
                   </button>
                   
                   {/* Zoom Controls */}
                   <div className="flex items-center gap-1 ml-2 bg-slate-900 border border-slate-800 rounded px-1 py-0.5">
                     <button 
                       onClick={() => setFollowPlayhead(!followPlayhead)}
                       className={`p-1 transition-colors ${followPlayhead ? 'text-amber-500' : 'text-slate-400 hover:text-white'}`}
                       title={followPlayhead ? "Disable Auto-scroll" : "Enable Auto-scroll"}
                     >
                       {followPlayhead ? <Link2 className="w-3.5 h-3.5" /> : <Link2Off className="w-3.5 h-3.5" />}
                     </button>
                     <div className="w-[1px] h-3 bg-slate-800"></div>
                     <button 
                       onClick={() => setShowShortcuts(true)}
                       className="p-1 text-slate-400 hover:text-white transition-colors"
                       title="Keyboard Shortcuts (?)"
                     >
                       <HelpCircle className="w-3.5 h-3.5" />
                     </button>
                     <div className="w-[1px] h-3 bg-slate-800"></div>
                     <button 
                       onClick={() => {
                         const container = timelineContainerRef.current;
                         if (!container) return;
                         const centerTime = (container.scrollLeft + container.offsetWidth / 2) / zoomLevel;
                         const newZoom = Math.max(2, zoomLevel / 1.25);
                         setZoomLevel(newZoom);
                         requestAnimationFrame(() => {
                           if (container) container.scrollLeft = centerTime * newZoom - container.offsetWidth / 2;
                         });
                       }}
                       className="p-1 hover:text-white transition-colors"
                       title="Zoom Out"
                     >
                        <X className="w-3 h-3 rotate-45" />
                     </button>
                     <div className="w-[1px] h-3 bg-slate-800"></div>
                     <button 
                       onClick={() => {
                         const container = timelineContainerRef.current;
                         if (!container) return;
                         const centerTime = (container.scrollLeft + container.offsetWidth / 2) / zoomLevel;
                         const newZoom = Math.min(500, zoomLevel * 1.25);
                         setZoomLevel(newZoom);
                         requestAnimationFrame(() => {
                           if (container) container.scrollLeft = centerTime * newZoom - container.offsetWidth / 2;
                         });
                       }}
                       className="p-1 hover:text-white transition-colors"
                       title="Zoom In"
                     >
                        <Play className="w-3 h-3 -rotate-90" />
                     </button>
                     <button 
                       onClick={() => {
                         const containerWidth = timelineContainerRef.current?.offsetWidth || 800;
                         const duration = Math.max(videoDuration, maxEndTime) || 1;
                         setZoomLevel(Math.max(2, (containerWidth - 40) / duration));
                       }}
                       className="p-1 px-1.5 hover:text-white transition-colors text-[9px] font-bold border-l border-slate-800"
                       title="Fit to Screen"
                     >
                        FIT
                     </button>
                     <span className="text-[9px] text-slate-500 font-mono ml-1 w-6 text-center">{Math.round(zoomLevel / 20 * 100)}%</span>
                   </div>
                 </div>
                 <span className="font-mono text-[10px] bg-slate-900 border border-slate-800 px-1.5 py-0.5 rounded text-slate-500">
                   {formatTime(currentTime)} / {formatTime(Math.max(videoDuration, subtitles.length > 0 ? Math.max(...subtitles.map(s => s.endTime)) : 0))}
                 </span>
              </h3>
              
              <div 
                ref={timelineContainerRef}
                className="flex-1 relative bg-slate-900 rounded border border-slate-800 overflow-x-auto overflow-y-hidden select-none custom-scrollbar pb-2"
                onWheel={(e) => {
                  if (e.altKey || e.ctrlKey || e.metaKey) {
                    e.preventDefault();
                    const zoomSensitivity = 0.05;
                    const direction = e.deltaY > 0 ? -1 : 1;
                    const zoomFactor = 1 + direction * zoomSensitivity;
                    
                    const container = timelineContainerRef.current;
                    if (!container) return;
                    
                    // Time position under the mouse
                    const rect = container.getBoundingClientRect();
                    const mouseXInContainer = e.clientX - rect.left + container.scrollLeft;
                    const timeAtMouse = mouseXInContainer / zoomLevel;
                    
                    const newZoom = Math.max(5, Math.min(500, zoomLevel * zoomFactor));
                    
                    if (newZoom !== zoomLevel) {
                      setZoomLevel(newZoom);
                      
                      // After state update, we need to adjust scroll
                      // Since setZoomLevel is async, we'll use a trick or just wait for the next render
                      // Better approach: Calculate the new scroll left immediately
                      const newScrollLeft = timeAtMouse * newZoom - (e.clientX - rect.left);
                      
                      // Use requestAnimationFrame to ensure the DOM has updated with new zoom-related widths
                      requestAnimationFrame(() => {
                        if (container) {
                          container.scrollLeft = newScrollLeft;
                        }
                      });
                    }
                  }
                }}
              >
                <div 
                   className="h-full relative min-w-full cursor-pointer"
                   style={{ width: `${Math.max(100, Math.max(videoDuration, subtitles.length > 0 ? Math.max(...subtitles.map(s => s.endTime)) : 0) * zoomLevel)}px` }} // Dynamic scaling
                   onPointerDown={(e) => {
                     setIsScrubbing(true);
                     isScrubbingRef.current = true;
                     e.currentTarget.setPointerCapture(e.pointerId);
                     if (videoRef.current) {
                       if (!videoRef.current.paused) {
                          videoRef.current.pause();
                          e.currentTarget.dataset.wasPlaying = 'true';
                       }
                       const rect = e.currentTarget.getBoundingClientRect();
                       const clickX = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
                       const percentage = clickX / rect.width;
                       const currentTotal = Math.max(videoDuration, maxEndTime) || 1;
                       handleTimelineScrub(percentage * currentTotal);
                     }
                   }}
                 onPointerMove={(e) => {
                   if (isScrubbingRef.current && videoRef.current) {
                     const rect = e.currentTarget.getBoundingClientRect();
                     const clickX = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
                     const percentage = clickX / rect.width;
                     let totalD = Math.max(videoDuration, subtitles.length > 0 ? Math.max(...subtitles.map(s => isNaN(s.endTime) ? 0 : s.endTime)) : 0) || 1;
                     if (isNaN(totalD)) totalD = 1;
                     const newTime = Math.max(0, Math.min(percentage * totalD, totalD));
                     videoRef.current.currentTime = newTime;
                     
                     // Force immediate audio scrub
                     if (currentAudioRef.current && activeSubtitleIdRef.current !== null) {
                        const activeSub = subtitles.find(s => s.id === activeSubtitleIdRef.current);
                        if (activeSub && newTime >= activeSub.startTime && newTime <= activeSub.endTime) {
                           if (Math.abs(currentAudioRef.current.currentTime - (newTime - activeSub.startTime)) > 0.05) {
                               currentAudioRef.current.currentTime = newTime - activeSub.startTime;
                           }
                        }
                     }
                   }
                 }}
                 onPointerUp={(e) => {
                   setIsScrubbing(false);
                   isScrubbingRef.current = false;
                   e.currentTarget.releasePointerCapture(e.pointerId);
                   if (videoRef.current && e.currentTarget.dataset.wasPlaying === 'true') {
                     videoRef.current.play().catch(console.error);
                     delete e.currentTarget.dataset.wasPlaying;
                   }
                 }}
              >
                {/* Background grid / ruler */}
                <div className="absolute top-0 left-0 right-0 h-6 border-b border-slate-800">
                  {timelineRuler}
                </div>

                {/* Video Track */}
                <div className="absolute top-8 left-0 right-0 h-10 flex">
                   <div 
                     className="bg-slate-800/80 border border-slate-700 rounded-sm relative overflow-hidden flex items-center pl-2 text-slate-600 pointer-events-none"
                     style={{ 
                       width: videoDuration ? `${(videoDuration / totalDuration) * 100}%` : '100%',
                       minWidth: '4px'
                     }}
                   >
                     {/* Thumbnail placeholders to look like a timeline track */}
                     <div className="absolute inset-0 flex opacity-20 object-cover pointer-events-none" style={{ backgroundImage: 'repeating-linear-gradient(90deg, transparent, transparent 40px, rgba(255,255,255,0.05) 40px, rgba(255,255,255,0.05) 42px)' }}></div>
                     <Film className="w-4 h-4 shrink-0 text-slate-500 mr-2 z-10" />
                     <span className="text-[10px] font-medium z-10 text-slate-400 truncate">{videoFile ? videoFile.name : 'Untitled Video Track'}</span>
                   </div>
                </div>

                {/* Subtitles & Audio Track */}
                <div className="absolute top-[76px] left-0 right-0 h-24">
                  {subtitles.map((sub) => (
                    <TimelineSubtitleItem 
                      key={sub.id}
                      sub={sub}
                      currentTime={currentTime}
                      totalDuration={totalDuration}
                      zoomLevel={zoomLevel}
                      draggingSub={draggingSub}
                      handleTimelineDrag={handleTimelineDrag}
                      handleTimelineDragEnd={handleTimelineDragEnd}
                      handleToggleLink={handleToggleLink}
                      videoRef={videoRef}
                      isActive={currentTime >= sub.startTime && currentTime <= sub.endTime}
                      isOverlapping={audioOverlaps.has(sub.id)}
                    />
                  ))}
                </div>

                {/* Playhead indicator */}
                <div 
                   className="absolute top-0 bottom-0 w-[2px] bg-amber-400 z-40 pointer-events-none shadow-[0_0_8px_rgba(251,191,36,0.5)]"
                   style={{ 
                     left: `${(currentTime / totalDuration) * 100}%` 
                   }}
                >
                   {/* Playhead handle */}
                   <div 
                      className="absolute top-0 left-1/2 -translate-x-1/2 bg-amber-400 text-amber-950 flex flex-col items-center justify-center shadow-lg"
                      style={{ 
                        width: '12px', 
                        height: '16px', 
                        clipPath: 'polygon(0 0, 100% 0, 100% 60%, 50% 100%, 0 60%)' 
                      }}
                   >
                     <div className="w-[1.5px] h-2 bg-amber-950/40 mt-1"></div>
                   </div>
                </div>
              </div>
            </div>
          </div>
        </div>

          {/* Mobile upload controls */}
          <div className="md:hidden p-4 border-t border-slate-800 bg-slate-900/50 flex flex-col gap-2 shrink-0">
             <div className="flex gap-2">
               <label className="flex-1 text-center py-2 px-4 bg-slate-800 border border-slate-700 rounded text-xs font-medium text-slate-300 cursor-pointer">
                 {videoFile ? 'Change Video' : 'Upload Video'}
                 <input type="file" accept="video/mp4,video/webm" onChange={handleVideoUpload} className="hidden" />
               </label>
               <label className="flex-1 text-center py-2 px-4 bg-slate-800 border border-slate-700 rounded text-xs font-medium text-slate-300 cursor-pointer">
                 {subtitles.length > 0 ? 'Change SRT' : 'Upload SRT'}
                 <input type="file" accept=".srt" onChange={handleSRTUpload} className="hidden" />
               </label>
             </div>
             <div className="flex gap-2">
               <label className="flex-1 text-center py-2 px-4 bg-slate-800 border border-slate-700 rounded text-xs font-medium text-slate-300 cursor-pointer">
                 {referenceAudioFile ? 'Change Ref Audio' : 'Upload Ref Audio'}
                 <input type="file" accept="audio/*" onChange={handleReferenceAudioUpload} className="hidden" />
               </label>
             </div>
             {subtitles.length > 0 && (
                <button
                  onClick={handleGenerateAll}
                  disabled={isGeneratingAll}
                  className="w-full py-2 bg-amber-500 text-slate-950 rounded text-xs font-bold disabled:opacity-50"
               >
                 {isGeneratingAll ? 'Generating...' : 'Generate All Audio'}
               </button>
             )}
             <div className="mt-4 flex flex-col gap-2">
               <label className="text-xs text-slate-500 uppercase font-bold">TTS Engine</label>
               <div className="flex flex-wrap bg-slate-800 rounded p-1 gap-1">
                 <button 
                  onClick={() => setTtsEngine('google-free')}
                  className={`flex-1 min-w-[70px] text-[10px] py-1.5 rounded transition bg-transparent ${ttsEngine === 'google-free' ? 'bg-slate-700 text-white' : 'text-slate-400 hover:text-white'}`}
                 >Google Free</button>
                 <button 
                  onClick={() => setTtsEngine('voxcpm')}
                  className={`flex-1 min-w-[70px] text-[10px] py-1.5 rounded transition ${ttsEngine === 'voxcpm' ? 'bg-purple-600 text-white font-medium' : 'text-slate-400 hover:text-white'}`}
                 >VoxCPM</button>
                 <button 
                  onClick={() => setTtsEngine('gemini')}
                  className={`flex-1 min-w-[70px] text-[10px] py-1.5 rounded transition ${ttsEngine === 'gemini' ? 'bg-amber-600 text-white font-medium' : 'text-slate-400 hover:text-white'}`}
                 >Gemini</button>
               </div>
               {ttsEngine === 'google-free' && (
                 <p className="text-[10px] text-slate-500">Free, basic text-to-speech. Unlimited attempts.</p>
               )}
               {ttsEngine === 'voxcpm' && (
                 <p className="text-[10px] text-purple-400/80">Uses local VoxCPM Gradio endpoint for generation.</p>
               )}
               {ttsEngine === 'gemini' && (
                 <p className="text-[10px] text-amber-500/70">High-quality dramatic voice. Limited by quota (15 RPM free).</p>
               )}
             </div>
          </div>
        </main>

        {/* Right Panel: Subtitle List */}
        <aside className="w-80 border-l border-slate-800 bg-slate-900/30 flex flex-col shrink-0 max-w-full">
          <div className="p-4 border-b border-slate-800 flex items-center justify-between shrink-0">
            <h3 className="text-xs font-bold text-slate-500 uppercase">អត្ថបទ SRT (Subtitles)</h3>
            <div className="flex gap-2">
              <button 
                onClick={() => setIsBatchEditMode(!isBatchEditMode)}
                className={`p-1.5 rounded transition ${isBatchEditMode ? 'bg-amber-500/20 text-amber-500' : 'hover:bg-slate-800 text-slate-500 hover:text-slate-300'}`}
                title="Batch Edit"
              >
                <ListChecks className="w-4 h-4" />
              </button>
              <button className="p-1 hover:bg-slate-800 rounded">
                 <Music className="w-4 h-4 text-slate-500" />
              </button>
            </div>
          </div>
          
          {isBatchEditMode && (
            <div className="p-3 border-b border-slate-800 bg-slate-800/20 text-xs flex flex-col gap-3 shrink-0">
              <div className="flex items-center justify-between">
                <div className="flex gap-3 items-center">
                  <button 
                    onClick={handleSelectAll}
                    className="text-amber-500 hover:text-amber-400 capitalize underline"
                  >
                    Select All
                  </button>
                  <span className="text-slate-500">|</span>
                  <button 
                    onClick={handleDeselectAll}
                    className="text-slate-400 hover:text-slate-300 capitalize underline"
                  >
                    None
                  </button>
                </div>
                <span className="text-slate-400">{selectedSubtitles.size} selected</span>
              </div>
              <div className="flex flex-col gap-2">
                <div className="text-[10px] text-slate-500 uppercase font-bold tracking-wider">Find & Replace</div>
                <input 
                  type="text" 
                  placeholder="Find text..." 
                  value={batchFindText}
                  onChange={(e) => setBatchFindText(e.target.value)}
                  className="bg-slate-900 border border-slate-700 rounded px-2 py-1.5 outline-none text-slate-200"
                />
                <div className="flex gap-2">
                  <input 
                    type="text" 
                    placeholder="Replace with..." 
                    value={batchReplaceText}
                    onChange={(e) => setBatchReplaceText(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1.5 outline-none text-slate-200"
                  />
                  <button 
                    onClick={handleBatchReplace}
                    className="bg-amber-500 hover:bg-amber-400 text-slate-900 px-3 rounded font-bold whitespace-nowrap transition-colors"
                  >
                    Replace
                  </button>
                </div>
              </div>
              <div className="flex flex-col gap-2">
                 <div className="text-[10px] text-slate-500 uppercase font-bold tracking-wider">Adjust Timing</div>
                 <div className="flex gap-2 items-center">
                   <input 
                     type="number"
                     step="0.1"
                     value={batchTimeShift}
                     onChange={(e) => setBatchTimeShift(e.target.value)}
                     className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1.5 outline-none text-slate-200"
                     placeholder="Shift seconds (e.g. 1.5, -2)"
                   />
                   <button 
                     onClick={handleBatchTimeShift}
                     className="bg-purple-600 hover:bg-purple-500 text-white px-3 py-1.5 rounded font-bold whitespace-nowrap transition-colors"
                   >
                     Shift Time
                   </button>
                 </div>
              </div>
            </div>
          )}

          {errorMsg && (
            <div className="p-3.5 m-3 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-[11px] font-medium animate-in fade-in slide-in-from-top-2 duration-300 relative group pr-10 shadow-lg shadow-red-950/20">
              <div className="flex items-start gap-3.5">
                <div className="mt-0.5 shrink-0 p-1.5 bg-red-500/10 rounded-lg border border-red-500/20">
                  <X className="w-3 h-3 text-red-400/80" />
                </div>
                <div className="flex flex-col gap-1.5 flex-1 pr-2">
                  <span className="font-bold flex items-center gap-2 text-red-300">
                    <span className="w-1 h-1 rounded-full bg-red-500" />
                    TTS Generation Issue
                  </span>
                  <span className="opacity-80 leading-relaxed break-words">{errorMsg}</span>
                  <div className="mt-1 flex items-center gap-3">
                    <button 
                      onClick={() => setErrorMsg('')}
                      className="text-[10px] font-bold text-red-400/60 hover:text-red-400 underline underline-offset-2 transition-colors uppercase tracking-wider"
                    >
                      Dismiss
                    </button>
                    {(errorMsg.includes('API Key') || errorMsg.includes('Settings')) && (
                      <span className="text-[9px] text-slate-500 italic">Hint: Go to Settings tab above</span>
                    )}
                  </div>
                </div>
              </div>
              <button 
                onClick={() => setErrorMsg('')}
                className="absolute top-3 right-3 p-1 rounded-md bg-transparent opacity-40 group-hover:opacity-100 transition-all hover:bg-red-500/20"
                title="Dismiss"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          )}

          <div className="flex-1 overflow-y-auto custom-scrollbar">
            {subtitles.length > 0 ? (
               <div className="flex flex-col">
                 {subtitles.map((sub) => (
                    <SubtitleListItem 
                      key={sub.id}
                      sub={sub}
                      isActive={currentTime >= sub.startTime && currentTime <= sub.endTime}
                      isBatchEditMode={isBatchEditMode}
                      selectedSubtitles={selectedSubtitles}
                      previewingId={previewingId}
                      isGeneratingAll={isGeneratingAll}
                      ttsEngine={ttsEngine}
                      defaultVoxCPMVoice={defaultVoxCPMVoice}
                      referenceAudioFile={referenceAudioFile}
                      referenceAudioBase64={referenceAudioBase64}
                      handleToggleLink={handleToggleLink}
                      toggleSubtitleSelection={toggleSubtitleSelection}
                      handlePreviewAudio={handlePreviewAudio}
                      handleGenerateSingle={handleGenerateSingle}
                      handleEngineChange={handleEngineChange}
                      handleVoiceChange={handleVoiceChange}
                      handleSubReferenceAudioUpload={handleSubReferenceAudioUpload}
                      playAudioFile={playAudioFile}
                      updateSubtitles={updateSubtitles}
                      videoRef={videoRef}
                    />
                  ))}
               </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-center text-slate-600 p-4">
                <FileText className="w-10 h-10 mb-3 opacity-20" />
                <p className="text-sm">Upload an SRT file to load subtitles and generate dramatic Khmer audio.</p>
              </div>
            )}
          </div>
        </aside>
      </div>

      {/* Settings Modal */}
      {showSettings && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-slate-900 border border-slate-700 rounded-xl w-full max-w-md p-6 shadow-2xl">
            <h2 className="text-xl font-semibold text-white mb-6">API Configuration</h2>
            
            <div className="space-y-4 mb-6 max-h-[65vh] overflow-y-auto pr-2 custom-scrollbar">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1.5 flex justify-between">
                  <span>VoxCPM Server URL & Voice</span>
                  <a href="https://support.google.com/chrome/answer/99020" target="_blank" rel="noreferrer" className="text-[10px] text-purple-400 hover:text-purple-300 underline">Mixed Content Fix</a>
                </label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={voxcpmUrl}
                    onChange={(e) => setVoxcpmUrl(e.target.value)}
                    placeholder="https://your-ngrok/v1/audio/speech"
                    className="flex-1 bg-slate-950 border border-slate-800 rounded-md px-3 py-2 text-sm text-white placeholder-slate-600 focus:outline-none focus:border-purple-500 focus:ring-1 focus:ring-purple-500 transition-all font-mono"
                  />
                  <select
                    value={defaultVoxCPMVoice}
                    onChange={(e) => setDefaultVoxCPMVoice(e.target.value)}
                    className="w-32 bg-slate-950 border border-slate-800 rounded-md px-2 py-2 text-sm text-white focus:outline-none focus:border-purple-500 transition-all"
                  >
                    {VOXCPM_VOICES.map((v) => (
                      <option key={v.id} value={v.id}>{v.label}</option>
                    ))}
                  </select>
                </div>
                <p className="text-[10px] text-slate-500 mt-1 leading-relaxed">
                  Full VoxCPM Gradio endpoint URL. Needs to be accessible from this app.
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1.5 flex justify-between">
                  <span>Gemini API Key & Default Voice</span>
                </label>
                <div className="flex gap-2">
                  <input
                    type="password"
                    value={geminiKey}
                    onChange={(e) => setGeminiKey(e.target.value)}
                    placeholder="AIza..."
                    className="flex-1 bg-slate-950 border border-slate-800 rounded-md px-3 py-2 text-sm text-white placeholder-slate-600 focus:outline-none focus:border-amber-500 focus:ring-1 focus:ring-amber-500 transition-all"
                  />
                  <select 
                    value={defaultGeminiVoice} 
                    onChange={(e) => setDefaultGeminiVoice(e.target.value)}
                    className="w-32 bg-slate-950 border border-slate-800 rounded-md px-2 py-2 text-sm text-white focus:outline-none focus:border-amber-500 transition-all"
                  >
                    {TTS_VOICES.map((v) => <option key={v.id} value={v.id}>{v.id}</option>)}
                  </select>
                </div>
              </div>

              <div className="pt-2 border-t border-slate-800">
              </div>
            </div>

            <div className="flex justify-end gap-3">
              <button
                onClick={() => setShowSettings(false)}
                className="px-4 py-2 bg-transparent hover:bg-slate-800 text-slate-300 rounded-md text-sm font-medium transition-colors"
               >
                Cancel
              </button>
              <button
                onClick={saveSettings}
                className="px-4 py-2 bg-amber-500 hover:bg-amber-400 text-slate-950 rounded-md text-sm font-bold shadow-lg shadow-amber-500/20 transition-all"
               >
                Save Changes
              </button>
            </div>
          </div>
        </div>
      )}

      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 8px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background-color: #334155;
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background-color: #475569;
        }
      `}</style>
    </div>
  );
}

