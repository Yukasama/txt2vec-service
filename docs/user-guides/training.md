# Training-Modul

Dieses Modul enthält die gesamte Logik für das Training von SBERT-Modellen (Sentence-BERT) inklusive Datenvalidierung, Trainingsorchestrierung, Fehlerbehandlung und Utilities für die Integration in REST-APIs.

## Trainingsprinzip

Das Training erfolgt mit **Triplet-Losses** basierend auf Triplets bestehend aus:
- **Anchor (Question)**: Ausgangsfrage oder Satz
- **Positive**: Semantisch ähnliche Antwort
- **Negative**: Semantisch unähnliche Antwort

**Ziel:** Die Embeddings von Anchor und Positive sollen möglichst ähnlich (hohe Kosinus-Ähnlichkeit) sein, während die Embeddings von Anchor und Negative möglichst unähnlich (niedrige Kosinus-Ähnlichkeit) werden.

Weitere Details siehe [SBERT-Dokumentation](https://www.sbert.net/docs/package_reference/losses.html).


## Übersicht

Das Trainingssystem ist modular aufgebaut und bietet:

- **Datenvalidierung**: Sicherstellung, dass Trainingsdaten korrekt und vollständig sind
- **Trainingsorchestrierung**: Kapselung des gesamten Trainingsablaufs (Vorbereitung, Training, Speichern, Aufräumen)
- **Fehlerbehandlung**: Klare Fehlerklassen für alle typischen Fehlerfälle
- **Integration**: Einfache Anbindung an FastAPI-Endpunkte und Datenbank

## Architektur

### Hauptkomponenten

1. **`tasks.py`**: Zentrale Trainingslogik (Background-Tasks)
   - Training-Tasks für asynchrone Ausführung
   - Integration mit Dramatiq für Task-Management
   - Fehlerbehandlung und Status-Updates

2. **`service.py`**: Service-Layer für API-Integration

   - Validierung von Requests
   - Aufruf der Training-Tasks
   - Fehler- und Statusmanagement

3. **`router.py`**: REST-API Endpunkte
   - `POST /train`: Training starten
   - `GET /{task_id}/status`: Status abfragen
   - HTTP Response-Handling

4. **`repository.py`**: Datenbankoperationen
   - CRUD-Operationen für Training-Tasks
   - Datenbankmodell-Management

5. **`utils/`**: Hilfsfunktionen
   - **`input_examples.py`**: Erstellung und Filterung von Trainingsbeispielen
   - **`model_loader.py`**: Laden und Initialisieren von Modellen
   - **`cleanup.py`**: Ressourcenmanagement und Aufräumarbeiten
   - **`safetensors_finder.py`**: Auffinden von Modell-Dateien

## Beispiel: Training per Python

```python
from vectorize.training.service import TrainingOrchestrator
from vectorize.training.schemas import TrainRequest
from sqlmodel.ext.asyncio.session import AsyncSession

# Training-Request erstellen
train_request = TrainRequest(
    model_tag="models--sentence-transformers--all-MiniLM-L6-v2",
    train_dataset_ids=["0b30b284-f7fe-4e6c-a270-17cafc5b5bcb"],
    epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
)

# Training-Orchestrator verwenden
db = AsyncSession()  # Ihre DB-Session
orchestrator = TrainingOrchestrator(db, task_id)

# Training starten (asynchron)
await orchestrator.run_training(
    model_path=model_path,
    train_request=train_request,
    dataset_paths=dataset_paths,
    output_dir=output_dir
)
```

## Beispiel: Training per REST-API

Das Training wird als Background-Task gestartet und liefert eine Task-ID zurück:

```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "model_tag": "models--sentence-transformers--all-MiniLM-L6-v2",
    "train_dataset_ids": ["0b30b284-f7fe-4e6c-a270-17cafc5b5bcb"],
    "epochs": 3,
    "per_device_train_batch_size": 16,
    "learning_rate": 2e-5,
    "warmup_steps": 100,
    "evaluation_steps": 500
  }'
```

Response:

```json
{
  "status_code": 202,
  "headers": {
    "Location": "/training/7ef54ba0-2d87-4864-8360-81de8035369a/status"
  }
}
```

### Status abfragen

```bash
curl "http://localhost:8000/training/{task_id}/status"
```

### Alle verfügbaren Parameter

Die API unterstützt alle wichtigen Sentence-Transformers Parameter:

- `model_tag`: Basis-Modell aus der Datenbank
- `train_dataset_ids`: Liste von Dataset-IDs (werden zusammengeführt)
- `val_dataset_id`: Optional, für Validierung
- `epochs`, `per_device_train_batch_size`, `learning_rate`
- `warmup_steps`, `optimizer_name`, `scheduler`
- `weight_decay`, `max_grad_norm`, `use_amp`
- `show_progress_bar`, `evaluation_steps`, `output_path`
- `save_best_model`, `save_each_epoch`, `save_optimizer_state`
- `dataloader_num_workers`, `device`, `timeout_seconds`

## Fehlerbehandlung

Das System bietet detaillierte Fehlerklassen für verschiedene Szenarien:

- **`InvalidDatasetIdError`**: Ungültige UUID oder fehlende Dataset-ID
- **`InvalidModelIdError`**: Ungültiges Modell oder Modell-Tag nicht gefunden
- **`TrainingDatasetNotFoundError`**: Trainingsdatensatz existiert nicht im Dateisystem
- **`TrainingModelWeightsNotFoundError`**: Modellgewichte nicht gefunden
- **`TrainingTaskNotFoundError`**: Training-Task mit gegebener ID nicht gefunden
- **`DatasetValidationError`**: Datensatz hat falsche Struktur oder fehlerhafte Daten
- **Timeout-Errors**: Training überschreitet das definierte Zeitlimit

Alle Fehler werden detailliert geloggt und über die API zurückgegeben.

## Vollständige JSON-API-Referenz

### Training Request Beispiele

**Minimal Training Request:**

```json
{
  "model_tag": "models--sentence-transformers--all-MiniLM-L6-v2",
  "train_dataset_ids": ["0b30b284-f7fe-4e6c-a270-17cafc5b5bcb"]
}
```

**Standard Training Request:**

```json
{
  "model_tag": "models--sentence-transformers--all-MiniLM-L6-v2",
  "train_dataset_ids": [
    "0b30b284-f7fe-4e6c-a270-17cafc5b5bcb",
    "0a9d5e87-e497-4737-9829-2070780d10df"
  ],
  "val_dataset_id": "0a9d5e87-e497-4737-9829-2070780d10df",
  "epochs": 3,
  "learning_rate": 0.00005,
  "per_device_train_batch_size": 8
}
```

**Vollständige Training Request mit allen Parametern:**

```json
{
  "model_tag": "models--sentence-transformers--all-MiniLM-L6-v2",
  "train_dataset_ids": [
    "0b30b284-f7fe-4e6c-a270-17cafc5b5bcb",
    "0a9d5e87-e497-4737-9829-2070780d10df"
  ],
  "val_dataset_id": "0a9d5e87-e497-4737-9829-2070780d10df",
  "epochs": 3,
  "per_device_train_batch_size": 16,
  "learning_rate": 5e-5,
  "warmup_steps": 200,
  "optimizer_name": "AdamW",
  "scheduler": "constantlr",
  "weight_decay": 0.01,
  "max_grad_norm": 1.0,
  "use_amp": true,
  "show_progress_bar": true,
  "evaluation_steps": 1000,
  "save_best_model": true,
  "save_each_epoch": false,
  "save_optimizer_state": false,
  "dataloader_num_workers": 0,
  "device": "cuda",
  "timeout_seconds": 7200
}
```

### Training Response Beispiele

**Training Start Response (202 Accepted):**

```json
{
  "status_code": 202,
  "headers": {
    "Location": "/training/7ef54ba0-2d87-4864-8360-81de8035369a/status"
  }
}
```

**Training Status Response:**

```json
{
  "task_id": "7ef54ba0-2d87-4864-8360-81de8035369a",
  "status": "DONE",
  "created_at": "2025-06-15T21:34:47.000Z",
  "end_date": "2025-06-15T22:15:23.000Z",
  "error_msg": null,
  "trained_model_id": "trained_models/models--sentence-transformers--all-MiniLM-L6-v2-finetuned-20250615-213447-7ef54ba0"
}
```

### Datenaufteilung und Validierungslogik

**Szenario 1: Mehrere Datasets mit expliziter Validierung**

```json
{
  "train_dataset_ids": ["dataset1-uuid", "dataset2-uuid"],
  "val_dataset_id": "validation-dataset-uuid"
}
```

→ **Ergebnis**: `validation-dataset-uuid` wird als Validierungsdatensatz verwendet

**Szenario 2: Mehrere Datasets ohne explizite Validierung**

```json
{
  "train_dataset_ids": ["dataset1-uuid", "dataset2-uuid"]
}
```

→ **Ergebnis**: System verknüpft alle Datensätze und teilt 90% Training / 10% Validierung

**Szenario 3: Einzelner Datensatz ohne Validierung**

```json
{
  "train_dataset_ids": ["single-dataset-uuid"]
}
```

→ **Ergebnis**: Auto-Split von `single-dataset-uuid` → 90% Training / 10% Validierung

### Datenpfad-Beispiele

**Training erstellt diese Pfade:**

- Explizite Validierung: `data/datasets/my_validation_dataset.jsonl`
- Auto-Split Validierung: `data/datasets/my_training_dataset.jsonl#auto-split`

### Parameter-Regeln

**Erforderlich:**

- `model_tag`: Immer erforderlich
- `train_dataset_ids`: Mindestens ein Dataset (Array mit min. 1 Element)

**Optional:**

- `val_dataset_id`: Für explizite Validierungsdaten
- Alle anderen Parameter haben Standardwerte

### Ungültige Kombinationen

**Fehlerhafte per_device_train_batch_size:**
```json
{
  "model_tag": "...",
  "train_dataset_ids": ["..."],
  "per_device_train_batch_size": 0 // Fehler: muss > 0 sein
}
```

**Fehlerhafte learning_rate:**

```json
{
  "model_tag": "...",
  "train_dataset_ids": ["..."],
  "learning_rate": 0 // Fehler: muss > 0 sein
}
```

## Integration mit Evaluation

Nach erfolgreichem Training kann das trainierte Modell direkt evaluiert werden:

**1. Training starten:**

```bash
POST /train
{
  "model_tag": "models--sentence-transformers--all-MiniLM-L6-v2",
  "train_dataset_ids": ["dataset-uuid"],
  "val_dataset_id": "validation-uuid",
  "epochs": 3
}
```

**2. Training-Task-ID aus Response extrahieren:**

```json
{
  "headers": {
    "Location": "/training/7ef54ba0-2d87-4864-8360-81de8035369a/status"
  }
}
```

**3. Trainiertes Modell evaluieren:**

```bash
POST /evaluation/evaluate
{
  "model_tag": "trained_models/models--sentence-transformers--all-MiniLM-L6-v2-finetuned-20250615-213447-7ef54ba0",
  "training_task_id": "7ef54ba0-2d87-4864-8360-81de8035369a",
  "baseline_model_tag": "models--sentence-transformers--all-MiniLM-L6-v2"
}
```

Diese Integration ermöglicht konsistente Evaluierung mit denselben Validierungsdaten, die beim Training verwendet wurden.

## Testen

Tests für das Trainingsmodul befinden sich in `tests/training/`:

```bash
pytest tests/training/ -v
```

- **Valid-Tests**: Erfolgreiches Training mit korrekten Daten
- **Invalid-Tests**: Fehlerfälle und fehlerhafte Daten

## Datenformat

Das System erwartet JSONL-Dateien mit folgender Struktur:

```json
{"question": "Was ist maschinelles Lernen?", "positive": "ML ist ein Teilbereich der KI", "negative": "Das Wetter ist heute schön"}
{"question": "Wie funktioniert Deep Learning?", "positive": "Mit neuronalen Netzen", "negative": "Pizza schmeckt gut"}
```

**Erforderliche Spalten:**

- `question`: Die Ausgangsfrage oder der Anchor
- `positive`: Semantisch ähnliche/korrekte Antwort
- `negative`: Semantisch unähnliche/falsche Antwort

**Validierung:**

- Alle Spalten müssen vorhanden sein
- Keine NULL-Werte oder leere Strings
- Mindestens ein Trainingseintrag erforderlich

## Leseliteratur

### Was sind Sentence-Transformers?

Sentence-Transformers sind spezialisierte Modelle, die auf BERT, RoBERTa oder ähnlichen Architekturen basieren und darauf trainiert sind, ganze Sätze oder Textabschnitte als dichte Vektoren (Embeddings) im semantischen Raum abzubilden. Dadurch können semantisch ähnliche Sätze durch ähnliche Vektoren repräsentiert werden.

---

Für Details siehe auch die Docstrings in den jeweiligen Modulen und die API-Dokumentation (Swagger/OpenAPI).
