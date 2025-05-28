'use client';

import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ChartData,
  ChartOptions,
  PointElement,
  LineElement,
} from 'chart.js';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Camera,
  Zap,
  Smile,
  Frown,
  Meh,
  Angry,
  AlertTriangle,
  Loader2,
  Info,
  Github,
  WifiOff,
  Wifi,
  Settings2,
  Palette,
} from 'lucide-react';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  PointElement,
  LineElement
);

// --- Interfaces ---
interface LandmarkPoint {
  x: number;
  y: number;
  z?: number;
}

interface FrameAnalysisData {
  landmarks?: LandmarkPoint[];
  dominant_emotion?: string;
  emotion_probabilities?: Record<string, number>;
  processing_time_ms: number;
  message?: string;
  error?: boolean;
  error_message?: string;
}

// --- Styling and Constants ---
const emotionStyles: Record<
  string,
  {
    icon: React.ElementType;
    color: string;
    emoji: string;
    landmarkColor?: string;
  }
> = {
  happy: {
    icon: Smile,
    color: 'text-green-400',
    emoji: '😊',
    landmarkColor: '#34D399',
  }, // Emerald 500
  sad: {
    icon: Frown,
    color: 'text-blue-400',
    emoji: '😢',
    landmarkColor: '#60A5FA',
  }, // Blue 400
  angry: {
    icon: Angry,
    color: 'text-red-500',
    emoji: '😠',
    landmarkColor: '#EF4444',
  }, // Red 500
  neutral: {
    icon: Meh,
    color: 'text-gray-400',
    emoji: '😐',
    landmarkColor: '#9CA3AF',
  }, // Gray 400
  surprise: {
    icon: Zap,
    color: 'text-yellow-400',
    emoji: '😮',
    landmarkColor: '#FACC15',
  }, // Yellow 400
  fear: {
    icon: Frown,
    color: 'text-purple-400',
    emoji: '😨',
    landmarkColor: '#A78BFA',
  }, // Purple 400
  disgust: {
    icon: Angry,
    color: 'text-lime-600',
    emoji: '🤢',
    landmarkColor: '#84CC16',
  }, // Lime 500 (distinguish from angry)

  // Fallback/Special states
  custom: {
    icon: Settings2,
    color: 'text-teal-400',
    emoji: '⚙️',
    landmarkColor: '#2DD4BF',
  }, // Teal 400
  prediction_error: {
    icon: AlertTriangle,
    color: 'text-orange-400',
    emoji: '⚠️',
    landmarkColor: '#F97316',
  },
  normalization_failed: {
    icon: AlertTriangle,
    color: 'text-yellow-500',
    emoji: '❓',
    landmarkColor: '#EAB308',
  },
  default: {
    icon: Palette,
    color: 'text-pink-400',
    emoji: '🎨',
    landmarkColor: '#F472B6',
  }, // Default if no emotion
};

const getEmotionStyle = (emotion?: string) => {
  if (!emotion) return emotionStyles.default;
  return emotionStyles[emotion.toLowerCase()] || emotionStyles.custom;
};

const FRAME_CAPTURE_INTERVAL_MS = 100; // Target ~10 FPS. Adjust based on performance.
const VIDEO_WIDTH = 640; // Logical capture width
const VIDEO_HEIGHT = 360; // Logical capture height
const LANDMARK_POINT_SIZE = 1.2; // Visual size of drawn landmark points
const LANDMARK_DEFAULT_COLOR = '#0EA5E9'; // Sky 500 - for when no specific emotion color

export default function EmotionDetectionPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null); // For capturing frames to send
  const wsRef = useRef<WebSocket | null>(null);
  const frameSendIntervalRef = useRef<NodeJS.Timeout | null>(null); // Renamed for clarity
  const animationFrameIdRef = useRef<number | null>(null); // For drawing loop
  const mediaStreamRef = useRef<MediaStream | null>(null);

  const [isWebcamOn, setIsWebcamOn] = useState(false);
  const [latestFrameData, setLatestFrameData] =
    useState<FrameAnalysisData | null>(null);
  const [isConnecting, setIsConnecting] = useState(false); // For WS connection attempt
  const [isSendingFrame, setIsSendingFrame] = useState(false); // UI feedback for frame send
  const [appError, setAppError] = useState<string | null>(null);
  const [backendMessage, setBackendMessage] = useState<string | null>(null);
  const [webSocketStatus, setWebSocketStatus] = useState<
    'disconnected' | 'connecting' | 'connected' | 'error'
  >('disconnected');

  // Use environment variable for WebSocket URL, fallback to localhost
  const WS_API_URL =
    process.env.NEXT_PUBLIC_API_WS_URL ||
    'ws://localhost:8000/ws/emotion_detection';

  // --- WebSocket Management ---
  const connectToWebSocket = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      logger.info('WebSocket already open.');
      setWebSocketStatus('connected');
      return;
    }
    setWebSocketStatus('connecting');
    setIsConnecting(true);
    wsRef.current = new WebSocket(WS_API_URL);

    wsRef.current.onopen = () => {
      logger.info('WebSocket connection established.');
      setWebSocketStatus('connected');
      setAppError(null);
      setIsConnecting(false);
      startSendingFrames(); // Begin sending frames
    };

    wsRef.current.onmessage = (event) => {
      setIsSendingFrame(false); // Reset after response
      try {
        const data: FrameAnalysisData = JSON.parse(event.data as string);
        setLatestFrameData(data);
        if (data.error) {
          logger.error('Error from backend:', data.error_message);
          setAppError(data.error_message || 'Backend processing error.');
        } else {
          if (data.message && !data.landmarks && !data.dominant_emotion) {
            setBackendMessage(data.message); // E.g., "No face detected"
          } else {
            setBackendMessage(null); // Clear message if data is present
          }
        }
      } catch (e) {
        logger.error('Failed to parse WebSocket message:', e);
        setAppError('Malformed data received from server.');
      }
    };

    wsRef.current.onerror = (errorEvent) => {
      logger.error('WebSocket error:', errorEvent);
      setAppError('WebSocket connection failed. Ensure backend is running.');
      setWebSocketStatus('error');
      setIsConnecting(false);
      setIsSendingFrame(false);
    };

    wsRef.current.onclose = (closeEvent) => {
      logger.info(
        `WebSocket closed. Code: ${closeEvent.code}, Reason: ${closeEvent.reason}`
      );
      if (webSocketStatus !== 'error') {
        // Don't override if it was an error close
        setWebSocketStatus('disconnected');
      }
      setIsConnecting(false);
      setIsSendingFrame(false);
      stopSendingFrames();
      setLatestFrameData(null);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [WS_API_URL, webSocketStatus]); // Added webSocketStatus

  const disconnectFromWebSocket = useCallback(() => {
    stopSendingFrames();
    if (wsRef.current) {
      wsRef.current.close(1000, 'Client initiated disconnect'); // Code 1000 for normal closure
      wsRef.current = null;
    }
    setWebSocketStatus('disconnected');
    setLatestFrameData(null);
  }, []); // Removed stopSendingFrames from dep array as it's defined below

  // --- Frame Capturing and Sending ---
  const startSendingFrames = useCallback(() => {
    if (frameSendIntervalRef.current)
      clearInterval(frameSendIntervalRef.current); // Clear existing
    frameSendIntervalRef.current = setInterval(() => {
      if (
        wsRef.current?.readyState === WebSocket.OPEN &&
        videoRef.current &&
        videoRef.current.readyState >= videoRef.current.HAVE_CURRENT_DATA &&
        !videoRef.current.paused
      ) {
        if (!offscreenCanvasRef.current) {
          offscreenCanvasRef.current = document.createElement('canvas');
          offscreenCanvasRef.current.width = VIDEO_WIDTH;
          offscreenCanvasRef.current.height = VIDEO_HEIGHT;
        }
        const context = offscreenCanvasRef.current.getContext('2d');
        if (context) {
          context.drawImage(videoRef.current, 0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);
          const dataUrl = offscreenCanvasRef.current.toDataURL(
            'image/jpeg',
            0.65
          ); // Quality 0.65
          wsRef.current.send(dataUrl);
          setIsSendingFrame(true);
        }
      }
    }, FRAME_CAPTURE_INTERVAL_MS);
  }, []);

  const stopSendingFrames = useCallback(() => {
    if (frameSendIntervalRef.current) {
      clearInterval(frameSendIntervalRef.current);
      frameSendIntervalRef.current = null;
    }
    setIsSendingFrame(false);
  }, []);

  // --- Webcam Management ---
  const activateWebcam = useCallback(async () => {
    setAppError(null);
    setBackendMessage(null);
    setIsConnecting(true); // Use isConnecting for webcam start too
    try {
      mediaStreamRef.current = await navigator.mediaDevices.getUserMedia({
        video: { width: VIDEO_WIDTH, height: VIDEO_HEIGHT, facingMode: 'user' },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStreamRef.current;
        await new Promise<void>((resolve) => {
          // Ensure video is ready
          if (videoRef.current)
            videoRef.current.onloadedmetadata = () => resolve();
          else resolve();
        });
        setIsWebcamOn(true);
        connectToWebSocket(); // Connect WebSocket after webcam is on
      }
    } catch (err: unknown) {
      if (err instanceof Error) {
        logger.error('Failed to start webcam:', err);
        setAppError(
          `Webcam access failed: ${err.message}. Please check browser permissions.`
        );
      } else {
        logger.error('Failed to start webcam:', err);
        setAppError('Webcam access failed due to an unknown error.');
      }
      setIsWebcamOn(false);
      setIsConnecting(false);
    }
  }, [connectToWebSocket]);

  const deactivateWebcam = useCallback(() => {
    disconnectFromWebSocket();
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    setIsWebcamOn(false);
    setLatestFrameData(null);
    setAppError(null);
    setBackendMessage(null);
    if (overlayCanvasRef.current) {
      // Clear overlay
      const ctx = overlayCanvasRef.current.getContext('2d');
      if (ctx)
        ctx.clearRect(
          0,
          0,
          overlayCanvasRef.current.width,
          overlayCanvasRef.current.height
        );
    }
  }, [disconnectFromWebSocket]);

  const handleToggleWebcam = useCallback(() => {
    if (isWebcamOn) deactivateWebcam();
    else activateWebcam();
  }, [isWebcamOn, activateWebcam, deactivateWebcam]);

  // --- Drawing Loop for Overlay Canvas ---
  useEffect(() => {
    const drawOverlay = () => {
      if (
        !isWebcamOn ||
        !overlayCanvasRef.current ||
        !videoRef.current ||
        !videoRef.current.videoWidth
      ) {
        if (animationFrameIdRef.current)
          cancelAnimationFrame(animationFrameIdRef.current);
        animationFrameIdRef.current = requestAnimationFrame(drawOverlay); // Keep trying if webcam is on
        return;
      }

      const overlayCtx = overlayCanvasRef.current.getContext('2d');
      if (!overlayCtx) return;

      // Ensure canvas display size matches video display size
      overlayCanvasRef.current.width = videoRef.current.clientWidth;
      overlayCanvasRef.current.height = videoRef.current.clientHeight;

      const displayWidth = videoRef.current.clientWidth;
      const displayHeight = videoRef.current.clientHeight;

      overlayCtx.clearRect(0, 0, displayWidth, displayHeight);

      if (latestFrameData?.landmarks && latestFrameData.landmarks.length > 0) {
        const currentEmotion = latestFrameData.dominant_emotion;
        const emotionStyle = getEmotionStyle(currentEmotion);
        overlayCtx.fillStyle =
          emotionStyle.landmarkColor || LANDMARK_DEFAULT_COLOR;

        // Draw all landmarks
        latestFrameData.landmarks.forEach((lm) => {
          // Landmarks (x,y) are normalized (0-1) relative to the *captured* frame dimensions (VIDEO_WIDTH, VIDEO_HEIGHT)
          // We need to map them to the *displayed* video dimensions.
          const drawX = lm.x * displayWidth;
          const drawY = lm.y * displayHeight;
          overlayCtx.beginPath();
          overlayCtx.arc(drawX, drawY, LANDMARK_POINT_SIZE, 0, 2 * Math.PI);
          overlayCtx.fill();
        });

        // Draw dominant emotion text (corrected for mirroring)
        if (currentEmotion) {
          const text = `${emotionStyle.emoji} ${currentEmotion}`;
          const fontSize = Math.max(
            12,
            Math.min(displayWidth, displayHeight) * 0.04
          ); // Responsive font size
          overlayCtx.font = `bold ${fontSize}px 'Inter', sans-serif`;

          const textMetrics = overlayCtx.measureText(text);
          const textWidth = textMetrics.width;
          const textHeight = fontSize * 1.2;
          const padding = fontSize * 0.3;

          // Position text at the visual top-left of the mirrored view
          const visualTextX = padding;
          const textY = padding + textHeight * 0.8;

          overlayCtx.save();
          overlayCtx.scale(-1, 1); // Flip context for text drawing

          overlayCtx.fillStyle = 'rgba(0, 0, 0, 0.75)'; // Text background
          overlayCtx.fillRect(
            -(visualTextX + textWidth + 2 * padding), // x in flipped context
            padding / 2, // y
            textWidth + 2 * padding, // width
            textHeight + padding // height
          );

          overlayCtx.fillStyle = emotionStyle.landmarkColor || '#FFFFFF'; // Text color
          overlayCtx.textAlign = 'left';
          overlayCtx.fillText(
            text,
            -(visualTextX + textWidth + padding),
            textY
          );

          overlayCtx.restore(); // Restore context
        }
      }
      animationFrameIdRef.current = requestAnimationFrame(drawOverlay);
    };

    if (isWebcamOn) {
      animationFrameIdRef.current = requestAnimationFrame(drawOverlay);
    } else {
      if (animationFrameIdRef.current)
        cancelAnimationFrame(animationFrameIdRef.current);
      // Clear canvas if webcam turns off
      const overlayCtx = overlayCanvasRef.current?.getContext('2d');
      if (overlayCtx && overlayCanvasRef.current) {
        overlayCtx.clearRect(
          0,
          0,
          overlayCanvasRef.current.width,
          overlayCanvasRef.current.height
        );
      }
    }

    return () => {
      // Cleanup animation frame on unmount or if webcam turns off
      if (animationFrameIdRef.current) {
        cancelAnimationFrame(animationFrameIdRef.current);
      }
    };
  }, [isWebcamOn, latestFrameData]);

  // Cleanup on component unmount
  useEffect(() => {
    return () => {
      deactivateWebcam(); // This handles stopping stream and WebSocket
    };
  }, [deactivateWebcam]);

  // --- Chart Configuration ---
  const emotionChartData = (): ChartData<'bar'> | null => {
    if (
      !latestFrameData ||
      !latestFrameData.dominant_emotion ||
      !latestFrameData.emotion_probabilities
    )
      return null;

    const labels = Object.keys(latestFrameData.emotion_probabilities);
    const data = Object.values(latestFrameData.emotion_probabilities).map(
      (value) => value * 100
    ); // Assumed to be 0-1 from backend

    const chartColors: Record<string, string> = {
      /* ... (same as before, ensure keys match emotion strings) ... */
      happy: 'rgba(52, 211, 153, 0.7)',
      sad: 'rgba(96, 165, 250, 0.7)',
      angry: 'rgba(239, 68, 68, 0.7)',
      neutral: 'rgba(156, 163, 175, 0.7)',
      surprise: 'rgba(250, 204, 21, 0.7)',
      fear: 'rgba(167, 139, 250, 0.7)',
      disgust: 'rgba(132, 204, 22, 0.7)',
      custom: 'rgba(20, 184, 166, 0.7)',
      prediction_error: 'rgba(249, 115, 22, 0.7)',
      normalization_failed: 'rgba(234, 179, 8, 0.7)',
      default: 'rgba(236, 72, 153, 0.7)',
    };

    return {
      labels,
      datasets: [
        {
          label: `Intensity (%)`,
          data,
          backgroundColor: labels.map(
            (label) =>
              chartColors[label.toLowerCase() as keyof typeof chartColors] ||
              chartColors.default
          ),
          borderColor: labels.map((label) =>
            (
              chartColors[label.toLowerCase() as keyof typeof chartColors] ||
              chartColors.default
            ).replace('0.7', '1')
          ),
          borderWidth: 1,
          borderRadius: 5,
        },
      ],
    };
  };
  const emotionChartOptions: ChartOptions<'bar'> = {
    responsive: true,
    maintainAspectRatio: false,
    indexAxis: 'y',
    scales: {
      x: {
        beginAtZero: true,
        max: 100,
        ticks: { color: '#cbd5e1', font: { family: "'Inter', sans-serif" } },
        grid: { color: 'rgba(203, 213, 225, 0.1)' },
      },
      y: {
        ticks: { color: '#cbd5e1', font: { family: "'Inter', sans-serif" } },
        grid: { display: false },
      },
    },
    plugins: {
      legend: { display: false },
      title: {
        display: true,
        text: 'Emotion Profile',
        color: '#e2e8f0',
        font: { size: 16, family: "'Inter', sans-serif", weight: 'bold' },
      },
      tooltip: {
        callbacks: {
          label: (ctx) =>
            `${ctx.label || ''}: ${Number(ctx.parsed.x).toFixed(1)}%`,
        },
      },
    },
  };
  const currentEmotionChartData = emotionChartData();

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 text-slate-100 flex flex-col items-center p-2 sm:p-4 md:p-6 font-inter">
      <header className="w-full max-w-5xl mb-4 md:mb-6 text-center">
        <motion.h1
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-3xl sm:text-4xl md:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-sky-400 to-cyan-300 mb-1 md:mb-2"
        >
          Live Facial Emotion Detector
        </motion.h1>
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="text-slate-400 text-xs sm:text-sm md:text-base"
        >
          Using MediaPipe landmarks & custom model concept.
        </motion.p>
        <a
          href="https://github.com/FidaHussain87/facial_emotion_detection"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center mt-2 text-xs text-sky-400 hover:text-sky-300 transition-colors"
        >
          <Github size={14} className="mr-1" /> Inspired by a previous project
        </a>
      </header>

      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-xl lg:max-w-2xl bg-slate-800 rounded-xl shadow-2xl p-3 sm:p-4 md:p-6 mb-4 md:mb-6"
      >
        <div className="relative aspect-[16/9] bg-slate-700 rounded-lg overflow-hidden mb-3 md:mb-4 border-2 border-slate-600">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="w-full h-full object-cover transform scale-x-[-1]"
            width={VIDEO_WIDTH}
            height={VIDEO_HEIGHT}
          />
          <canvas
            ref={overlayCanvasRef}
            className="absolute top-0 left-0 w-full h-full pointer-events-none transform scale-x-[-1]"
          />

          {!isWebcamOn && !isConnecting && (
            <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-400 bg-slate-700 bg-opacity-80">
              <Camera size={48} className="mb-2 opacity-60" />{' '}
              <p className="text-sm">Webcam is Off</p>
            </div>
          )}
          {(isConnecting ||
            (isWebcamOn &&
              isSendingFrame &&
              !latestFrameData?.landmarks &&
              webSocketStatus === 'connected')) && (
            <div className="absolute inset-0 bg-black bg-opacity-60 flex flex-col items-center justify-center z-10">
              <Loader2 size={36} className="text-sky-400 animate-spin" />
              <p className="mt-2 text-sm text-sky-300">
                {isConnecting && webSocketStatus === 'connecting'
                  ? 'Connecting to Server...'
                  : isConnecting
                  ? 'Starting Webcam...'
                  : isSendingFrame
                  ? 'Analyzing...'
                  : 'Loading...'}
              </p>
            </div>
          )}
        </div>
        <div className="flex flex-col sm:flex-row items-center justify-between gap-3 md:gap-4">
          <button
            onClick={handleToggleWebcam}
            disabled={isConnecting && webSocketStatus === 'connecting'}
            className={`w-full sm:w-auto px-4 py-2 md:px-6 md:py-3 rounded-lg font-semibold text-white transition-all duration-300 ease-in-out flex items-center justify-center gap-2 text-sm md:text-base ${
              isWebcamOn
                ? 'bg-red-600 hover:bg-red-700'
                : 'bg-sky-500 hover:bg-sky-600'
            } focus:outline-none focus:ring-2 ${
              isWebcamOn ? 'focus:ring-red-400' : 'focus:ring-sky-400'
            } focus:ring-opacity-50 shadow-md hover:shadow-lg transform hover:-translate-y-0.5 disabled:opacity-60 disabled:cursor-not-allowed`}
          >
            <Camera size={18} />{' '}
            {isConnecting && webSocketStatus !== 'connected'
              ? 'Initializing...'
              : isWebcamOn
              ? 'Stop Webcam'
              : 'Start Webcam'}
          </button>
          <div className="flex items-center gap-2 text-xs text-slate-400">
            {webSocketStatus === 'connected' ? (
              <Wifi size={14} className="text-green-400" />
            ) : webSocketStatus === 'error' ||
              webSocketStatus === 'disconnected' ? (
              <WifiOff size={14} className="text-red-400" />
            ) : (
              <Loader2 size={14} className="text-yellow-400 animate-spin" />
            )}
            <span className="hidden sm:inline">
              {webSocketStatus.charAt(0).toUpperCase() +
                webSocketStatus.slice(1)}
            </span>
            {isWebcamOn && latestFrameData?.processing_time_ms != null && (
              <motion.div
                className="ml-1 sm:ml-2"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                Ping: {latestFrameData.processing_time_ms.toFixed(0)} ms
              </motion.div>
            )}
          </div>
        </div>
        {appError && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-3 p-2 md:p-3 bg-red-700 bg-opacity-30 text-red-300 border border-red-600 rounded-md flex items-center gap-2 text-xs sm:text-sm"
          >
            <AlertTriangle size={18} /> {appError}
          </motion.div>
        )}
        {backendMessage && !appError && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-3 p-2 md:p-3 bg-sky-700 bg-opacity-30 text-sky-300 border border-sky-600 rounded-md flex items-center gap-2 text-xs sm:text-sm"
          >
            <Info size={18} /> {backendMessage}
          </motion.div>
        )}
      </motion.div>

      <AnimatePresence>
        {isWebcamOn &&
          latestFrameData?.dominant_emotion &&
          latestFrameData?.emotion_probabilities &&
          currentEmotionChartData && (
            <motion.section
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.5 }}
              className="w-full max-w-xl lg:max-w-2xl"
            >
              <div className="bg-slate-800 p-3 sm:p-4 md:p-6 rounded-xl shadow-xl border border-slate-700">
                <div className="flex items-center mb-3 md:mb-4">
                  <div
                    className={`p-2 rounded-full mr-3 bg-opacity-20 ${getEmotionStyle(
                      latestFrameData.dominant_emotion
                    ).color.replace('text-', 'bg-')}`}
                  >
                    {React.createElement(
                      getEmotionStyle(latestFrameData.dominant_emotion).icon,
                      {
                        size: 24,
                        className: getEmotionStyle(
                          latestFrameData.dominant_emotion
                        ).color,
                      }
                    )}
                  </div>
                  <div>
                    <h3
                      className={`text-xl md:text-2xl font-semibold ${
                        getEmotionStyle(latestFrameData.dominant_emotion).color
                      }`}
                    >
                      {latestFrameData.dominant_emotion
                        .charAt(0)
                        .toUpperCase() +
                        latestFrameData.dominant_emotion.slice(1)}
                      <span className="ml-1">
                        {
                          getEmotionStyle(latestFrameData.dominant_emotion)
                            .emoji
                        }
                      </span>
                    </h3>
                    <p className="text-slate-400 text-xs sm:text-sm">
                      Confidence:{' '}
                      {(
                        latestFrameData.emotion_probabilities[
                          latestFrameData.dominant_emotion.toLowerCase()
                        ] * 100 || 0
                      ).toFixed(1)}
                      %
                    </p>
                  </div>
                </div>
                <div className="h-56 sm:h-64 md:h-72 w-full">
                  <Bar
                    options={emotionChartOptions}
                    data={currentEmotionChartData}
                  />
                </div>
              </div>
            </motion.section>
          )}
      </AnimatePresence>
      {!isWebcamOn && !latestFrameData && !appError && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="text-center text-slate-500 mt-6 md:mt-8"
        >
          <Info size={20} className="mx-auto mb-1" />
          <p className="text-sm">Start webcam for live emotion analysis.</p>
        </motion.div>
      )}
      <footer className="mt-8 md:mt-12 text-center text-xs text-slate-500">
        <p>
          &copy; {new Date().getFullYear()} Well Enhanced Emotion Analyzer. For
          educational and demonstration purposes.
        </p>
        <p>Facial models may have biases. Use responsibly.</p>
      </footer>
    </div>
  );
}

// Basic logger utility
const logger = {
  info: (...args: unknown[]) => console.log('[INFO]', ...args),
  error: (...args: unknown[]) => console.error('[ERROR]', ...args),
  warn: (...args: unknown[]) => console.warn('[WARN]', ...args),
};
