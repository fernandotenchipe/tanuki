import { useMemo, useState } from 'react'

const HOURS_OPTIONS = Array.from({ length: 13 }, (_, i) => ({ label: `${i + 1}H`, value: `${i + 1}H` }))

const VISUAL_STYLE_OPTIONS = [
  { label: 'Realistic', value: 'Realista' },
  { label: 'Artistic / painterly', value: 'Artístico / pictórico' },
  { label: 'Illustrated', value: 'Ilustrado' },
  { label: 'Abstract', value: 'Abstracto' },
  { label: 'Random / auto', value: 'Aleatorio / auto' },
]

const TEMA_OPTIONS = [
  { label: 'Portrait / person', value: 'Persona / retrato' },
  { label: 'Pet / animal', value: 'Mascota / animal' },
  { label: 'Landscape / environment', value: 'Paisaje / entorno' },
  { label: 'Object / symbol', value: 'Objeto / símbolo' },
  { label: 'Abstract forms', value: 'Formas abstractas' },
  { label: 'Random / auto', value: 'Aleatorio / auto' },
]

const PALETA_OPTIONS = [
  { label: 'Warm colors', value: 'Colores cálidos' },
  { label: 'Cool colors', value: 'Colores fríos' },
  { label: 'Neutrals', value: 'Neutros' },
  { label: 'Black and white', value: 'Blanco y negro' },
  { label: 'Random / auto', value: 'Aleatorio / auto' },
]

const DETALLE_OPTIONS = [
  { label: 'Simple', value: 'Sencillo' },
  { label: 'Balanced', value: 'Equilibrado' },
  { label: 'Highly detailed', value: 'Muy detallado' },
  { label: 'Random / auto', value: 'Aleatorio / auto' },
]

const FONDO_OPTIONS = [
  { label: 'Flat', value: 'Liso' },
  { label: 'Textured', value: 'Con textura' },
  { label: 'Scene / environment', value: 'Escena / ambiente' },
  { label: 'Random / auto', value: 'Aleatorio / auto' },
]

const LUZ_OPTIONS = [
  { label: 'Soft', value: 'Suave' },
  { label: 'Dramatic', value: 'Dramática' },
  { label: 'Natural', value: 'Natural' },
  { label: 'Random / auto', value: 'Aleatorio / auto' },
]

const LIBERTAD_OPTIONS = [
  { label: 'Very strict to request', value: 'Muy fiel a lo pedido' },
  { label: 'Artistic interpretation', value: 'Interpretación artística' },
  { label: 'Creative / free', value: 'Creativa / libre' },
  { label: 'Random / auto', value: 'Aleatorio / auto' },
]

const FILTRO_OPTIONS = [
  { label: 'Kawaii / cute', value: 'Kawaii / tierno' },
  { label: 'Anime / JP illustration', value: 'Anime / ilustración japonesa' },
  { label: 'Realistic', value: 'Realista' },
  { label: 'Dark / scary', value: 'Oscuro / de miedo' },
  { label: 'Sad / melancholic', value: 'Triste / melancólico' },
  { label: 'Deep / introspective', value: 'Profundo / introspectivo' },
  { label: 'Bright / cheerful', value: 'Alegre / luminoso' },
  { label: 'Mysterious', value: 'Misterioso' },
  { label: 'Epic / cinematic', value: 'Épico / cinematográfico' },
  { label: 'Surreal / dreamy', value: 'Surreal / onírico' },
  { label: 'Random / auto', value: 'Aleatorio / auto' },
]

const DEFAULT_VALUES = {
  tanukiName: '',
  disconnectTime: '',
  sleepHoursFeelNew: '',
  age: '',
  allowStepsAccess: '', // "Sí" | "No"
  styleVisual: '',
  tema: '',
  paleta: '',
  detalle: '',
  fondo: '',
  iluminacion: '',
  libertad: '',
  filtro: '',
}

export default function TanukiSurveyModal({ open, onClose, onSubmit }) {
  const [values, setValues] = useState(DEFAULT_VALUES)

  const handleClose = () => {
    setValues(DEFAULT_VALUES)
    onClose()
  }

  const update = (key) => (e) => setValues((p) => ({ ...p, [key]: e.target.value }))

  const canSubmit = useMemo(() => {
    return (
      values.tanukiName.trim() &&
      values.disconnectTime.trim() &&
      values.sleepHoursFeelNew.trim() &&
      String(values.age).trim() &&
      values.allowStepsAccess.trim() &&
      values.styleVisual.trim() &&
      values.tema.trim() &&
      values.paleta.trim() &&
      values.detalle.trim() &&
      values.fondo.trim() &&
      values.iluminacion.trim() &&
      values.libertad.trim() &&
      values.filtro.trim()
    )
  }, [values])

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!canSubmit) return

    const payload = {
      ...values,
      age: Number(values.age),
      createdAt: new Date().toISOString(),
    }

    // guarda local (para que “se quede” en tu proyecto en front)
    localStorage.setItem('tanuki_survey', JSON.stringify(payload))

    onSubmit?.(payload)
    handleClose()
  }

  if (!open) return null

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2 className="modal-title">Survey</h2>
          <button className="modal-close" type="button" onClick={handleClose}>
            ✕
          </button>
        </div>

        <form className="modal-body survey" onSubmit={handleSubmit}>
          {/* 1) Tanuki name */}
          <div className="survey-block">
            <label className="survey-label">What should your tanuki call you?</label>
            <input
              className="survey-input"
              type="text"
              placeholder="e.g., Alex"
              value={values.tanukiName}
              onChange={update('tanukiName')}
            />
          </div>

          {/* 2) Wind-down time */}
          <div className="survey-block">
            <label className="survey-label">When do you usually start winding down? (e.g., 8:00PM)</label>
            <input
              className="survey-input"
              type="text"
              placeholder="e.g., 8:00PM"
              value={values.disconnectTime}
              onChange={update('disconnectTime')}
            />
          </div>

          {/* 3) Hours of sleep to feel refreshed */}
          <div className="survey-block">
            <label className="survey-label">How many hours of sleep make you feel refreshed?</label>

            <div className="survey-radio-list">
              {HOURS_OPTIONS.map((opt) => (
                <label key={opt.value} className="survey-radio">
                  <input
                    type="radio"
                    name="sleepHoursFeelNew"
                    value={opt.value}
                    checked={values.sleepHoursFeelNew === opt.value}
                    onChange={update('sleepHoursFeelNew')}
                  />
                  <span>{opt.label}</span>
                </label>
              ))}

              <label className="survey-radio">
                <input
                  type="radio"
                  name="sleepHoursFeelNew"
                  value="Otro"
                  checked={values.sleepHoursFeelNew === 'Otro'}
                  onChange={update('sleepHoursFeelNew')}
                />
                <span>Other</span>
              </label>
            </div>
          </div>

          {/* 4) Age */}
          <div className="survey-block">
            <label className="survey-label">Enter your age</label>
            <input
              className="survey-input"
              type="number"
              min="1"
              max="120"
              placeholder="e.g., 21"
              value={values.age}
              onChange={update('age')}
            />
          </div>

          {/* 5) Allow steps access */}
          <div className="survey-block">
            <label className="survey-label">Allow the tanuki to read your daily steps?</label>

            <div className="survey-radio-row">
              {[
                { label: 'Yes', value: 'Sí' },
                { label: 'No', value: 'No' },
              ].map((opt) => (
                <label key={opt.value} className="survey-radio">
                  <input
                    type="radio"
                    name="allowStepsAccess"
                    value={opt.value}
                    checked={values.allowStepsAccess === opt.value}
                    onChange={update('allowStepsAccess')}
                  />
                  <span>{opt.label}</span>
                </label>
              ))}
            </div>
          </div>

          {/* 6) Visual style */}
          <div className="survey-block">
            <label className="survey-label">Visual style</label>
            <div className="survey-radio-list">
              {VISUAL_STYLE_OPTIONS.map((opt) => (
                <label key={opt.value} className="survey-radio">
                  <input
                    type="radio"
                    name="styleVisual"
                    value={opt.value}
                    checked={values.styleVisual === opt.value}
                    onChange={update('styleVisual')}
                  />
                  <span>{opt.label}</span>
                </label>
              ))}
            </div>
          </div>

          {/* 7) Main theme */}
          <div className="survey-block">
            <label className="survey-label">Main theme</label>
            <div className="survey-radio-list">
              {TEMA_OPTIONS.map((opt) => (
                <label key={opt.value} className="survey-radio">
                  <input
                    type="radio"
                    name="tema"
                    value={opt.value}
                    checked={values.tema === opt.value}
                    onChange={update('tema')}
                  />
                  <span>{opt.label}</span>
                </label>
              ))}
            </div>
          </div>

          {/* 8) Color palette */}
          <div className="survey-block">
            <label className="survey-label">Color palette</label>
            <div className="survey-radio-list">
              {PALETA_OPTIONS.map((opt) => (
                <label key={opt.value} className="survey-radio">
                  <input
                    type="radio"
                    name="paleta"
                    value={opt.value}
                    checked={values.paleta === opt.value}
                    onChange={update('paleta')}
                  />
                  <span>{opt.label}</span>
                </label>
              ))}
            </div>
          </div>

          {/* 9) Detail level */}
          <div className="survey-block">
            <label className="survey-label">Detail level</label>
            <div className="survey-radio-list">
              {DETALLE_OPTIONS.map((opt) => (
                <label key={opt.value} className="survey-radio">
                  <input
                    type="radio"
                    name="detalle"
                    value={opt.value}
                    checked={values.detalle === opt.value}
                    onChange={update('detalle')}
                  />
                  <span>{opt.label}</span>
                </label>
              ))}
            </div>
          </div>

          {/* 10) Background */}
          <div className="survey-block">
            <label className="survey-label">Background</label>
            <div className="survey-radio-list">
              {FONDO_OPTIONS.map((opt) => (
                <label key={opt.value} className="survey-radio">
                  <input
                    type="radio"
                    name="fondo"
                    value={opt.value}
                    checked={values.fondo === opt.value}
                    onChange={update('fondo')}
                  />
                  <span>{opt.label}</span>
                </label>
              ))}
            </div>
          </div>

          {/* 11) Lighting */}
          <div className="survey-block">
            <label className="survey-label">Lighting</label>
            <div className="survey-radio-list">
              {LUZ_OPTIONS.map((opt) => (
                <label key={opt.value} className="survey-radio">
                  <input
                    type="radio"
                    name="iluminacion"
                    value={opt.value}
                    checked={values.iluminacion === opt.value}
                    onChange={update('iluminacion')}
                  />
                  <span>{opt.label}</span>
                </label>
              ))}
            </div>
          </div>

          {/* 12) Creative freedom */}
          <div className="survey-block">
            <label className="survey-label">Creative freedom</label>
            <div className="survey-radio-list">
              {LIBERTAD_OPTIONS.map((opt) => (
                <label key={opt.value} className="survey-radio">
                  <input
                    type="radio"
                    name="libertad"
                    value={opt.value}
                    checked={values.libertad === opt.value}
                    onChange={update('libertad')}
                  />
                  <span>{opt.label}</span>
                </label>
              ))}
            </div>
          </div>

          {/* 13) Atmosphere / mood filter */}
          <div className="survey-block">
            <label className="survey-label">Atmosphere / mood filter</label>
            <div className="survey-radio-list">
              {FILTRO_OPTIONS.map((opt) => (
                <label key={opt.value} className="survey-radio">
                  <input
                    type="radio"
                    name="filtro"
                    value={opt.value}
                    checked={values.filtro === opt.value}
                    onChange={update('filtro')}
                  />
                  <span>{opt.label}</span>
                </label>
              ))}
            </div>
          </div>

          <div className="survey-actions">
            <button type="button" className="survey-btn ghost" onClick={handleClose}>
              Cancel
            </button>
            <button type="submit" className="survey-btn" disabled={!canSubmit}>
              Save
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
