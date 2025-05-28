'use client';

import React, { useState, useRef, useCallback, useEffect } from 'react';
// Removed Webcam from 'react-webcam' as we'll use direct video element and mediaDevices
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
} from 'lucide-react';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

interface FaceRegion {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface EmotionPredictionData {
  // Renamed from EmotionPrediction to avoid conflict with component name
  region: FaceRegion;
  dominant_emotion: string;
  emotion_probabilities: Record<string, number>;
}

interface WebSocketResponse {
  predictions: EmotionPredictionData[];
  processing_time_ms: number;
  message?: string;
  error?: boolean;
  error_message?: string;
}

const emotionStyles: Record<
  string,
  { icon: React.ElementType; color: string; emoji: string }
> = {
  happy: { icon: Smile, color: 'text-green-400', emoji: '😊' },
  sad: { icon: Frown, color: 'text-blue-400', emoji: '😢' },
  angry: { icon: Angry, color: 'text-red-500', emoji: '😠' },
  neutral: { icon: Meh, color: 'text-gray-400', emoji: '😐' },
  fear: { icon: Frown, color: 'text-purple-400', emoji: '😨' },
  disgust: { icon: Angry, color: 'text-red-700', emoji: '🤢' },
  surprise: { icon: Zap, color: 'text-yellow-400', emoji: '😮' },
  default: { icon: Meh, color: 'text-gray-500', emoji: '🤔' },
};

const getEmotionStyle = (emotion: string) => {
  return emotionStyles[emotion.toLowerCase()] || emotionStyles.default;
};

const FRAME_CAPTURE_INTERVAL_MS = 150; // Approx 6-7 FPS. Adjust for performance. (e.g. 100ms for 10FPS, 200ms for 5FPS)
const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 360;

export default function EmotionDetectionPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null); // For drawing bounding boxes
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null); // For capturing frames
  const wsRef = useRef<WebSocket | null>(null);
  const frameCaptureIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [detectedEmotions, setDetectedEmotions] = useState<
    EmotionPredictionData[]
  >([]);
  const [isLoading, setIsLoading] = useState(false); // For initial connection/setup
  const [isProcessingFrame, setIsProcessingFrame] = useState(false); // For per-frame processing indication
  const [error, setError] = useState<string | null>(null);
  const [lastApiMessage, setLastApiMessage] = useState<string | null>(null);
  const [processingTime, setProcessingTime] = useState<number>(0);
  const [socketStatus, setSocketStatus] = useState<
    'disconnected' | 'connecting' | 'connected' | 'error'
  >('disconnected');

  const API_WS_URL =
    process.env.NEXT_PUBLIC_API_WS_URL ||
    'ws://localhost:8000/ws/emotion_detection';

  const connectWebSocket = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      logger.info('WebSocket already connected.');
      setSocketStatus('connected');
      return;
    }

    setSocketStatus('connecting');
    setIsLoading(true);
    wsRef.current = new WebSocket(API_WS_URL);

    wsRef.current.onopen = () => {
      logger.info('WebSocket connected.');
      setSocketStatus('connected');
      setError(null);
      setIsLoading(false);
      startFrameCapture(); // Start sending frames once connected
    };

    wsRef.current.onmessage = (event) => {
      setIsProcessingFrame(false); // Reset processing frame flag
      try {
        const data: WebSocketResponse = JSON.parse(event.data as string);
        if (data.error) {
          logger.error('Backend error:', data.error_message);
          setError(data.error_message || 'Error from backend analysis.');
          setDetectedEmotions([]);
        } else {
          setDetectedEmotions(data.predictions);
          setProcessingTime(data.processing_time_ms);
          if (data.message && data.predictions.length === 0) {
            setLastApiMessage(data.message);
          } else {
            setLastApiMessage(null);
          }
        }
      } catch (e) {
        logger.error('Failed to parse WebSocket message:', e);
        setError('Received malformed data from server.');
      }
    };

    wsRef.current.onerror = (errEvent) => {
      logger.error('WebSocket error:', errEvent);
      setError(
        'WebSocket connection error. Is the backend server running and accessible?'
      );
      setSocketStatus('error');
      setIsLoading(false);
      setIsProcessingFrame(false);
    };

    wsRef.current.onclose = () => {
      logger.info('WebSocket disconnected.');
      if (socketStatus !== 'error') {
        // Don't override error state if it was an error-driven close
        setSocketStatus('disconnected');
      }
      setIsLoading(false);
      setIsProcessingFrame(false);
      stopFrameCapture(); // Ensure frame capture stops if WS closes
      // Optionally, you can try to reconnect here with some strategy
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [API_WS_URL, socketStatus]); // Added socketStatus to dependencies, startFrameCapture removed as it's defined below

  const startFrameCapture = useCallback(() => {
    if (frameCaptureIntervalRef.current) {
      clearInterval(frameCaptureIntervalRef.current); // Clear existing interval if any
    }
    frameCaptureIntervalRef.current = setInterval(() => {
      if (
        wsRef.current?.readyState === WebSocket.OPEN &&
        videoRef.current &&
        !videoRef.current.paused &&
        !videoRef.current.ended
      ) {
        if (!offscreenCanvasRef.current) {
          // Create offscreen canvas if it doesn't exist
          offscreenCanvasRef.current = document.createElement('canvas');
          offscreenCanvasRef.current.width = VIDEO_WIDTH;
          offscreenCanvasRef.current.height = VIDEO_HEIGHT;
        }
        const context = offscreenCanvasRef.current.getContext('2d');
        if (context) {
          context.drawImage(videoRef.current, 0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);
          const dataUrl = offscreenCanvasRef.current.toDataURL(
            'image/jpeg',
            0.7
          ); // Send JPEG at 70% quality
          wsRef.current.send(dataUrl);
          setIsProcessingFrame(true); // Indicate that a frame has been sent
        }
      }
    }, FRAME_CAPTURE_INTERVAL_MS);
  }, []);

  const stopFrameCapture = useCallback(() => {
    if (frameCaptureIntervalRef.current) {
      clearInterval(frameCaptureIntervalRef.current);
      frameCaptureIntervalRef.current = null;
    }
    setIsProcessingFrame(false);
  }, []);

  const disconnectWebSocket = useCallback(() => {
    stopFrameCapture();
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null; // Ensure it's cleaned up
    }
    setSocketStatus('disconnected');
    setDetectedEmotions([]);
  }, [stopFrameCapture]);

  const startWebcam = useCallback(async () => {
    setError(null);
    setLastApiMessage(null);
    setIsLoading(true);
    try {
      streamRef.current = await navigator.mediaDevices.getUserMedia({
        video: { width: VIDEO_WIDTH, height: VIDEO_HEIGHT, facingMode: 'user' },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = streamRef.current;
        // Use a promise to wait for onloadedmetadata
        await new Promise<void>((resolve) => {
          if (videoRef.current) {
            videoRef.current.onloadedmetadata = () => {
              resolve();
            };
          } else {
            resolve(); // Should not happen if videoRef.current is true
          }
        });
        setIsWebcamActive(true);
        connectWebSocket(); // Connect WebSocket after webcam is successfully started
      }
    } catch (err: unknown) {
      if (err instanceof Error) {
        logger.error('Error starting webcam:', err);
        setError(
          `Failed to access webcam: ${err.message}. Please check permissions.`
        );
      } else {
        logger.error('Unknown error starting webcam:', err);
        setError(
          'Failed to access webcam due to an unknown error. Please check permissions.'
        );
      }
      setIsWebcamActive(false);
      setIsLoading(false);
    }
  }, [connectWebSocket]);

  const stopWebcam = useCallback(() => {
    disconnectWebSocket();
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsWebcamActive(false);
    setDetectedEmotions([]);
    setError(null);
    setLastApiMessage(null);
    if (overlayCanvasRef.current) {
      const ctx = overlayCanvasRef.current.getContext('2d');
      if (ctx)
        ctx.clearRect(
          0,
          0,
          overlayCanvasRef.current.width,
          overlayCanvasRef.current.height
        );
    }
  }, [disconnectWebSocket]);

  const toggleWebcam = useCallback(() => {
    if (isWebcamActive) {
      stopWebcam();
    } else {
      startWebcam();
    }
  }, [isWebcamActive, startWebcam, stopWebcam]);

  // Effect for drawing bounding boxes on overlay canvas
  useEffect(() => {
    if (
      !isWebcamActive ||
      !overlayCanvasRef.current ||
      !videoRef.current ||
      !videoRef.current.videoWidth
    )
      return;

    const overlayCtx = overlayCanvasRef.current.getContext('2d');
    if (!overlayCtx) return;

    overlayCanvasRef.current.width = videoRef.current.clientWidth;
    overlayCanvasRef.current.height = videoRef.current.clientHeight;

    const scaleX = videoRef.current.clientWidth / VIDEO_WIDTH;
    const scaleY = videoRef.current.clientHeight / VIDEO_HEIGHT;

    overlayCtx.clearRect(
      0,
      0,
      overlayCanvasRef.current.width,
      overlayCanvasRef.current.height
    );

    if (detectedEmotions.length > 0) {
      detectedEmotions.forEach((prediction) => {
        const { x, y, w, h } = prediction.region;
        const style = getEmotionStyle(prediction.dominant_emotion);

        const drawX = x * scaleX;
        const drawY = y * scaleY;
        const drawW = w * scaleX;
        const drawH = h * scaleY;

        const colorMap: Record<string, string> = {
          green: '#34D399',
          blue: '#60A5FA',
          red: '#F87171',
          gray: '#9CA3AF',
          purple: '#A78BFA',
          yellow: '#FBBF24',
          white: '#FFFFFF',
        };
        let emotionColorKey = 'gray'; // Default
        if (style.color.includes('green')) emotionColorKey = 'green';
        else if (style.color.includes('blue')) emotionColorKey = 'blue';
        else if (style.color.includes('red')) emotionColorKey = 'red';
        else if (style.color.includes('purple')) emotionColorKey = 'purple';
        else if (style.color.includes('yellow')) emotionColorKey = 'yellow';

        overlayCtx.strokeStyle = colorMap[emotionColorKey] || '#FFFFFF';
        overlayCtx.lineWidth = 2;
        overlayCtx.strokeRect(drawX, drawY, drawW, drawH);

        // Text Drawing
        overlayCtx.font = `bold ${
          14 * Math.min(scaleX, scaleY)
        }px 'Inter', sans-serif`;
        const emotionText = `${style.emoji} ${prediction.dominant_emotion}`;
        const textMetrics = overlayCtx.measureText(emotionText);
        const textWidth = textMetrics.width;

        const textBackgroundHeight = 14 * Math.min(scaleX, scaleY) + 8 * scaleY; // Font size + vertical padding
        const textPaddingX = 5 * scaleX;

        const backgroundRectX = drawX;
        const backgroundRectY = drawY - textBackgroundHeight - 4 * scaleY; // Position above box
        const backgroundRectWidth = textWidth + 2 * textPaddingX;

        overlayCtx.fillStyle = 'rgba(0, 0, 0, 0.65)'; // Background for text
        overlayCtx.fillRect(
          backgroundRectX,
          backgroundRectY,
          backgroundRectWidth,
          textBackgroundHeight
        );

        overlayCtx.fillStyle = overlayCtx.strokeStyle; // Text color same as box stroke

        // --- Text Rendering Fix ---
        overlayCtx.save();
        overlayCtx.scale(-1, 1); // Flip context horizontally for this drawing operation

        const visualTextLeftX = backgroundRectX + textPaddingX;
        const visualTextBaselineY =
          backgroundRectY + textBackgroundHeight - 4 * scaleY; // Adjust for baseline

        overlayCtx.textAlign = 'left'; // Ensure text draws left-to-right in the flipped context
        // To make text appear starting at visualTextLeftX (left edge in final view) and read LTR:
        // Draw at -(visualTextLeftX + textWidth) in the flipped context.
        // This means the text's right edge in the flipped context is at -visualTextLeftX.
        overlayCtx.fillText(
          emotionText,
          -(visualTextLeftX + textWidth),
          visualTextBaselineY
        );

        overlayCtx.restore(); // Restore context
        // --- End Text Rendering Fix ---
      });
    }
  }, [detectedEmotions, isWebcamActive]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopWebcam();
    };
  }, [stopWebcam]);

  const chartData = (prediction: EmotionPredictionData): ChartData<'bar'> => {
    const labels = Object.keys(prediction.emotion_probabilities);
    const data = Object.values(prediction.emotion_probabilities).map(
      (p) => p
    );
    const chartColors: Record<string, string> = {
      happy: 'rgba(74, 222, 128, 0.7)',
      sad: 'rgba(96, 165, 250, 0.7)',
      angry: 'rgba(239, 68, 68, 0.7)',
      neutral: 'rgba(156, 163, 175, 0.7)',
      fear: 'rgba(167, 139, 250, 0.7)',
      disgust: 'rgba(185, 28, 28, 0.7)',
      surprise: 'rgba(250, 204, 21, 0.7)',
    };
    return {
      labels,
      datasets: [
        {
          label: 'Emotion Intensity (%)',
          data,
          backgroundColor: labels.map(
            (label) =>
              chartColors[label.toLowerCase() as keyof typeof chartColors] ||
              'rgba(107, 114, 128, 0.7)'
          ),
          borderColor: labels.map(
            (label) =>
              chartColors[
                label.toLowerCase() as keyof typeof chartColors
              ]?.replace('0.7', '1') || 'rgba(107, 114, 128, 1)'
          ),
          borderWidth: 1,
          borderRadius: 5,
        },
      ],
    };
  };
  const chartOptions: ChartOptions<'bar'> = {
    responsive: true,
    maintainAspectRatio: false,
    indexAxis: 'y',
    scales: {
      x: {
        beginAtZero: true,
        max: 100,
        ticks: { color: '#cbd5e1', font: { family: "'Inter', sans-serif" } },
        grid: { color: 'rgba(203, 213, 225, 0.2)' },
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
        text: 'Emotion Probabilities',
        color: '#e2e8f0',
        font: { size: 16, family: "'Inter', sans-serif", weight: 'bold' },
      },
      tooltip: {
        callbacks: {
          label: (ctx) =>
            `${ctx.dataset.label || ''}: ${Number(ctx.parsed.x).toFixed(2)}%`,
        },
      },
    },
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 text-slate-100 flex flex-col items-center p-4 sm:p-6 font-inter">
      <header className="w-full max-w-5xl mb-6 text-center">
        <motion.h1
          className="text-4xl sm:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-sky-400 to-cyan-300 mb-2"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          Live Facial Emotion Detector
        </motion.h1>
        <motion.p
          className="text-slate-400 text-sm sm:text-base"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          Enhanced with WebSocket for smoother, real-time analysis.
        </motion.p>
        <a
          href="https://github.com/FidaHussain87/facial_emotion_detection"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center mt-3 text-xs text-sky-400 hover:text-sky-300 transition-colors"
        >
          <Github size={14} className="mr-1" /> Inspired by a previous project
        </a>
      </header>

      <motion.div
        className="w-full max-w-2xl bg-slate-800 rounded-xl shadow-2xl p-4 sm:p-6 mb-6"
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        <div className="relative aspect-video bg-slate-700 rounded-lg overflow-hidden mb-4 border-2 border-slate-700">
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

          {!isWebcamActive && !isLoading && (
            <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-400">
              <Camera size={64} className="mb-4 opacity-50" />
              <p>Webcam is off.</p>
            </div>
          )}
          {(isLoading ||
            (isWebcamActive &&
              isProcessingFrame &&
              detectedEmotions.length === 0 &&
              socketStatus === 'connected')) && (
            <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center z-10">
              <Loader2 size={48} className="text-sky-400 animate-spin" />
              <p className="ml-2 text-sky-300">
                {isLoading
                  ? socketStatus === 'connecting'
                    ? 'Connecting...'
                    : 'Starting webcam...'
                  : 'Processing...'}
              </p>
            </div>
          )}
        </div>

        <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
          <button
            onClick={toggleWebcam}
            disabled={isLoading && socketStatus === 'connecting'}
            className={`w-full sm:w-auto px-6 py-3 rounded-lg font-semibold text-white transition-all duration-300 ease-in-out flex items-center justify-center gap-2 ${
              isWebcamActive
                ? 'bg-red-500 hover:bg-red-600'
                : 'bg-sky-500 hover:bg-sky-600'
            } focus:outline-none focus:ring-2 ${
              isWebcamActive ? 'focus:ring-red-400' : 'focus:ring-sky-400'
            } focus:ring-opacity-50 shadow-md hover:shadow-lg transform hover:-translate-y-0.5 disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            <Camera size={20} />{' '}
            {isLoading && socketStatus !== 'connected'
              ? 'Starting...'
              : isWebcamActive
              ? 'Stop Webcam'
              : 'Start Webcam'}
          </button>
          <div className="flex items-center gap-2 text-xs text-slate-400">
            {socketStatus === 'connected' ? (
              <Wifi size={16} className="text-green-400" />
            ) : socketStatus === 'error' || socketStatus === 'disconnected' ? (
              <WifiOff size={16} className="text-red-400" />
            ) : (
              <Loader2 size={16} className="text-yellow-400 animate-spin" />
            )}
            <span>
              {socketStatus.charAt(0).toUpperCase() + socketStatus.slice(1)}
            </span>
            {isWebcamActive && processingTime > 0 && (
              <motion.div
                className="ml-2"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                Analysis: {processingTime.toFixed(0)} ms
              </motion.div>
            )}
          </div>
        </div>
        {error && (
          <motion.div
            className="mt-4 p-3 bg-red-500 bg-opacity-20 text-red-300 border border-red-500 rounded-md flex items-center gap-2"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <AlertTriangle size={20} /> {error}
          </motion.div>
        )}
        {lastApiMessage && !error && (
          <motion.div
            className="mt-4 p-3 bg-sky-500 bg-opacity-20 text-sky-300 border border-sky-500 rounded-md flex items-center gap-2"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <Info size={20} /> {lastApiMessage}
          </motion.div>
        )}
      </motion.div>

      <AnimatePresence>
        {isWebcamActive && detectedEmotions.length > 0 && (
          <motion.section
            className="w-full max-w-4xl grid grid-cols-1 md:grid-cols-2 gap-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.5 }}
          >
            {detectedEmotions.map((prediction, index) => {
              const style = getEmotionStyle(prediction.dominant_emotion);
              const dominantProb =
                prediction.emotion_probabilities[
                  prediction.dominant_emotion.toLowerCase()
                ] || 0;
              return (
                <motion.div
                  key={index}
                  className="bg-slate-800 p-4 sm:p-6 rounded-xl shadow-xl border border-slate-700"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <div className="flex items-center mb-4">
                    <style.icon size={36} className={`${style.color} mr-3`} />
                    <div>
                      <h3 className={`text-2xl font-semibold ${style.color}`}>
                        {prediction.dominant_emotion.charAt(0).toUpperCase() +
                          prediction.dominant_emotion.slice(1)}{' '}
                        {style.emoji}
                      </h3>
                      <p className="text-slate-400 text-sm">
                        Confidence: {dominantProb.toFixed(1)}%
                      </p>
                    </div>
                  </div>
                  <div className="h-64 sm:h-72 w-full">
                    <Bar options={chartOptions} data={chartData(prediction)} />
                  </div>
                </motion.div>
              );
            })}
          </motion.section>
        )}
      </AnimatePresence>
      {!isWebcamActive && detectedEmotions.length === 0 && !error && (
        <motion.div
          className="text-center text-slate-500 mt-8"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
        >
          <Info size={24} className="mx-auto mb-2" />
          <p>Start the webcam to see live emotion analysis.</p>
        </motion.div>
      )}
      <footer className="mt-12 text-center text-xs text-slate-500">
        <p>
          &copy; {new Date().getFullYear()} Well Enhanced Emotion Analyzer. For
          educational and demonstration purposes.
        </p>
        <p>Facial detection models may have biases. Use responsibly.</p>
      </footer>
    </div>
  );
}

const logger = {
  info: (...args: unknown[]) => console.log('[INFO]', ...args),
  error: (...args: unknown[]) => console.error('[ERROR]', ...args),
  warn: (...args: unknown[]) => console.warn('[WARN]', ...args),
};
