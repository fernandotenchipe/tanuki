import { useState } from 'react'
import './App.css'
import TanukiSurveyModal from './components/TanukiSurveyModal'
import tanuki from './assets/tanuki.png'
import tanukiGif from './assets/tanuki_fianl.gif'

function App() {
  const [steps, setSteps] = useState(8432)
  const [sleepHours, setSleepHours] = useState(7)
  const [sleepMinutes, setSleepMinutes] = useState(23)

  // modal
  const [isModalOpen, setIsModalOpen] = useState(false)

  // backend state
  const [isSending, setIsSending] = useState(false)
  const [lastResponse, setLastResponse] = useState(null)
  const [error, setError] = useState(null)
  const [isTraining, setIsTraining] = useState(false)
  const [trainMsg, setTrainMsg] = useState(null)
  const [trainSamples, setTrainSamples] = useState(200)
  const [trainEpochs, setTrainEpochs] = useState(80)
  const [isGenImg, setIsGenImg] = useState(false)
  const [imgB64, setImgB64] = useState(null)
  const [imgPrompt, setImgPrompt] = useState(null)

  const BACKEND_URL = 'http://localhost:8000/generate'
  const DATASET_URL = 'http://localhost:8000/make-dataset'
  const TRAIN_URL = 'http://localhost:8000/train'
  const IMAGE_URL = 'http://localhost:8000/generate-image'
  const IMAGE_STREAM_URL = 'http://localhost:8000/generate-image-stream'


  const handleSurveySubmit = async (surveyData) => {
    setIsSending(true)
    setError(null)
    setLastResponse(null)
    setImgB64(null)
    setImgPrompt(null)

    const payloadCommon = {
      steps: Number(steps),
      sleepHours: Number(sleepHours),
      sleepMinutes: Number(sleepMinutes),
      drawingStyle: surveyData?.styleVisual || 'Abstracto',
      survey: surveyData,
      createdAt: new Date().toISOString(),
    }

    try {
      // Generar SVG (receta)
      const res = await fetch(BACKEND_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payloadCommon),
      })

      if (!res.ok) {
        const txt = await res.text()
        throw new Error(`Backend ${res.status}: ${txt}`)
      }

      const data = await res.json()
      console.log('Respuesta backend:', data)
      setLastResponse(data)

      // Generar imagen (difusión) en una sola llamada (sin streaming para evitar errores de modulo)
      setIsGenImg(true)
      setImgB64(null)
      setImgPrompt(null)

      const imgRes = await fetch(IMAGE_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...payloadCommon,
          num_inference_steps: 35,
          guidance_scale: 8.5,
          width: 512,
          height: 512,
          trace: false,
        }),
      })

      if (!imgRes.ok) throw new Error(await imgRes.text())
      const imgData = await imgRes.json()
      if (!imgData.ok) throw new Error(imgData.error || 'Error generando imagen')
      setImgB64(imgData.image)
      setImgPrompt(imgData.prompt || null)
    } catch (err) {
      console.error(err)
      setError(err.message || 'Error desconocido')
    } finally {
      setIsGenImg(false)
      setIsSending(false)
    }
  }

  const handleTrain = async () => {
    setIsTraining(true)
    setTrainMsg(null)
    setError(null)

    // limpiar imagen al entrenar
    setImgB64(null)
    setImgPrompt(null)

    try {
      const samples = Math.max(10, Number(trainSamples) || 0)
      const epochs = Math.max(1, Number(trainEpochs) || 0)

      // 1) pedir dataset
      const dsRes = await fetch(DATASET_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ samples }),
      })
      if (!dsRes.ok) throw new Error(await dsRes.text())
      const ds = await dsRes.json()

      // 2) entrenar
      const trRes = await fetch(`${TRAIN_URL}?epochs=${epochs}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(ds.pairs),
      })
      if (!trRes.ok) throw new Error(await trRes.text())
      const tr = await trRes.json()

      setTrainMsg(tr)
      console.log('Train result:', tr)
    } catch (e) {
      setError(e.message || 'Error entrenando')
    } finally {
      setIsTraining(false)
    }
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <h1 className="app-title">Nemuri Tanuki</h1>
        <p className="app-subtitle">Your sleep companion</p>
      </header>

      <main className="app-main">
        <aside className="sidebar-panel">
          {/* Pasos */}
          <div className="param-card">
            <div className="param-icon">🚶</div>
            <div className="param-info">
              <label className="param-label">Daily Steps</label>
              <input
                className="param-input"
                type="number"
                min="0"
                value={steps}
                onChange={(e) => setSteps(e.target.value)}
              />
            </div>
          </div>

          {/* Sueño */}
          <div className="param-card">
            <div className="param-icon">💤</div>
            <div className="param-info">
              <label className="param-label">Sleep Data</label>

              <div className="sleep-inputs">
                <input
                  className="param-input sleep-num"
                  type="number"
                  min="0"
                  max="24"
                  value={sleepHours}
                  onChange={(e) => setSleepHours(e.target.value)}
                />
                <span className="sleep-sep">:</span>
                <input
                  className="param-input sleep-num"
                  type="number"
                  min="0"
                  max="59"
                  value={sleepMinutes}
                  onChange={(e) => setSleepMinutes(e.target.value)}
                />
              </div>

              <div className="sleep-hint">HH : MM</div>
            </div>
          </div>

          {/* Modal */}
          <button
            type="button"
            className="param-card param-card--mint"
            onClick={() => setIsModalOpen(true)}
            disabled={isSending || isTraining || isGenImg}
          >
            <div className="param-icon">📝</div>
            <div className="param-info">
              <span className="param-label">Survey</span>
              <span className="param-value">
                {isSending ? 'Sending…' : 'Open'}
              </span>
            </div>
          </button>

          {/* ✅ C) Entrenar IA */}
          <button
            type="button"
            className="param-card"
            onClick={handleTrain}
            disabled={isTraining || isSending || isGenImg}
          >
            <div className="param-icon">🧠</div>
            <div className="param-info">
              <span className="param-label">Train AI</span>
              <span className="param-value">
                {isTraining ? 'Training…' : 'Start'}
              </span>
            </div>
          </button>

          <div className="train-config">
            <div className="train-field">
              <span className="param-label">Samples</span>
              <input
                className="param-input train-input"
                type="number"
                min="10"
                max="2000"
                value={trainSamples}
                onChange={(e) => setTrainSamples(e.target.value)}
                disabled={isTraining || isSending || isGenImg}
              />
            </div>
            <div className="train-field">
              <span className="param-label">Epochs</span>
              <input
                className="param-input train-input"
                type="number"
                min="1"
                max="1000"
                value={trainEpochs}
                onChange={(e) => setTrainEpochs(e.target.value)}
                disabled={isTraining || isSending || isGenImg}
              />
            </div>
          </div>

          {/* Avatar */}
          <div className="icon-space">
            <div className="icon-circle">
              <img src={tanuki} alt="Tanuki" className="icon-logo-img" />
            </div>
            <span className="icon-label">Tanuki</span>
          </div>
        </aside>

        {/* PANEL PRINCIPAL */}
        <section className="main-panel">
          <div className="main-content">
            {error && (
              <span className="placeholder-text">{`Error: ${error}`}</span>
            )}

            {!error && imgB64 && (
              <div style={{ marginTop: 16, width: '100%', textAlign: 'center' }}>
                <img
                  src={`data:image/png;base64,${imgB64}`}
                  alt="Generada"
                  style={{ maxWidth: '880px', width: '100%', borderRadius: 16, boxShadow: '0 14px 32px rgba(0,0,0,0.28)' }}
                />
              </div>
            )}

            {!error && isGenImg && !imgB64 && (
              <div style={{ marginTop: 16, width: '100%', textAlign: 'center' }}>
                <img
                  src={tanukiGif}
                  alt="Generando…"
                  style={{ maxWidth: '560px', width: '100%', borderRadius: 16, boxShadow: '0 12px 30px rgba(0,0,0,0.18)' }}
                />
              </div>
            )}

            {!error && !imgB64 && !isGenImg && (
              <span className="placeholder-text">
                {isSending || isGenImg ? 'Generating image…' : 'Send the survey to view the final image'}
              </span>
            )}
          </div>
        </section>
      </main>

      <footer className="app-footer">
        <div className="color-dot cayenne"></div>
        <div className="color-dot weathered"></div>
        <div className="color-dot bitter"></div>
      </footer>

      {/* MODAL */}
      <TanukiSurveyModal
        open={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onSubmit={handleSurveySubmit}
      />
    </div>
  )
}

export default App
