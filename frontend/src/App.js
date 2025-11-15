import React, { useState, useRef } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [recording, setRecording] = useState(false);
  const [audioURL, setAudioURL] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const mediaRecorderRef = useRef(null);
  const audioChunks = useRef([]);

  const startRecording = async () => {
    setResult(null);
    setAudioURL(null);
    setError(null);
    audioChunks.current = [];

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const options = { mimeType: "audio/webm;codecs=opus" };
      const mediaRecorder = new MediaRecorder(stream, options);

      mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          audioChunks.current.push(e.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(audioChunks.current, { type: "audio/webm" });
        const url = URL.createObjectURL(blob);
        setAudioURL(url);
        stream.getTracks().forEach((t) => t.stop());
      };

      mediaRecorder.start();
      mediaRecorderRef.current = mediaRecorder;
      setRecording(true);

      setTimeout(() => {
        if (mediaRecorder.state !== "inactive") {
          mediaRecorder.stop();
          setRecording(false);
        }
      }, 5000);
    } catch (err) {
      setError("Microphone access denied. Please allow microphone access.");
      console.error(err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
      setRecording(false);
    }
  };

  const sendAudio = async () => {
    if (!audioURL) {
      setError("Record audio first (5 seconds).");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const blob = await fetch(audioURL).then(r => r.blob());
      const formData = new FormData();
      formData.append("audio", blob, "recording.webm");

      const res = await axios.post("http://localhost:8000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setResult(res.data);
    } catch (error) {
      console.error("FULL ERROR >>>", error);
      const errorMsg = error.response?.data?.error || error.message || "Unknown error occurred";
      setError(`Backend Error: ${errorMsg}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="stars"></div>
      <div className="stars2"></div>
      
      <div className="main-card">
        <div className="header">
          <h1 className="title">
            <span className="icon">🎤</span> Parkinson's Voice Detector
          </h1>
          <p className="subtitle">Advanced AI-powered voice analysis for early detection</p>
        </div>

        <div className="instruction-box">
          <span className="info-icon">ℹ️</span>
          <p>Say a sustained <strong>"aaaaaa"</strong> for about 4–5 seconds when recording starts.</p>
        </div>

        {error && (
          <div className="error-box">
            <span className="error-icon">❌</span>
            <p>{error}</p>
          </div>
        )}

        <div className="button-group">
          {!recording ? (
            <button className="btn btn-primary" onClick={startRecording}>
              <span className="btn-icon">🎙️</span> Start Recording
            </button>
          ) : (
            <button className="btn btn-danger recording-pulse" onClick={stopRecording}>
              <span className="pulse-dot"></span> Stop Recording
            </button>
          )}
        </div>

        {audioURL && (
          <div className="audio-section">
            <div className="audio-header">
              <span className="audio-icon">🔊</span>
              <h3>Your Recording</h3>
            </div>
            <audio controls className="audio-player" src={audioURL}></audio>
            <button 
              className="btn btn-success" 
              onClick={sendAudio} 
              disabled={loading}
            >
              <span className="btn-icon">{loading ? "⏳" : "📤"}</span>
              {loading ? "Processing..." : "Send for Analysis"}
            </button>
          </div>
        )}

        {result && (
          <div className={`result-section result-${result.prediction === 1 ? 'parkinsons' : 'healthy'}`}>
            <div className="result-header">
              <span className="result-icon">
                {result.prediction === 1 ? "🔴" : "🟢"}
              </span>
              <h2>Analysis Results</h2>
            </div>

            <div className="result-grid">
              <div className="result-item">
                <label>Status</label>
                <div className={`status-badge ${result.prediction === 1 ? 'positive' : 'negative'}`}>
                  {result.prediction === 1 ? "⚠️ Parkinson's Detected" : "✅ Healthy"}
                </div>
              </div>

              <div className="result-item">
                <label>Confidence</label>
                <div className="probability">
                  <div className="prob-bar">
                    <div 
                      className="prob-fill" 
                      style={{ width: `${(result.probability * 100)}%` }}
                    ></div>
                  </div>
                  <span className="prob-text">{(result.probability * 100).toFixed(2)}%</span>
                </div>
              </div>

              <div className="result-item full-width">
                <label>Assessment</label>
                <p className="message">{result.message}</p>
              </div>
            </div>

            {result.meta && (
              <div className="meta-info">
                <div className="meta-item">
                  <span className="meta-label">Duration:</span>
                  <span className="meta-value">{result.meta.duration?.toFixed(2)}s</span>
                </div>
                <div className="meta-item">
                  <span className="meta-label">Sample Rate:</span>
                  <span className="meta-value">{result.meta.samplerate} Hz</span>
                </div>
              </div>
            )}
          </div>
        )}

        <div className="footer">
          <p>Built with React & Machine Learning</p>
        </div>
      </div>
    </div>
  );
}

export default App;
