## Project Overview

This project is a modern, full-featured web application for managing, training, evaluating, and visualizing machine learning models, datasets, and related tasks. It is built with Next.js (React), TypeScript, Tailwind CSS, and leverages React Query for data fetching and caching. The UI is component-driven, modular, and designed for extensibility and maintainability.

---

## Key Features

### 1. **Model Management**
- Users can view, search, and interact with machine learning models.
- Each model has a detail page, accessible via `/model/[id]`, displaying metadata, evaluation results, and related actions.
- Existence checks for models are performed using React Query, ensuring robust error handling and user feedback.

### 2. **Dataset Management**
- Datasets are listed and can be viewed in detail.
- Dataset cards are rendered with clear, concise information, and long IDs are truncated for readability.
- The UI gracefully handles missing or unlinked datasets, displaying a clear message when no dataset is associated.

### 3. **Evaluation and Comparison**
- The evaluation detail view (`EvaluationData`) displays the status, progress, and results of model evaluations.
- Both baseline and target models are shown, with clickable cards if the resource exists, or a visually distinct error card if not.
- All date displays use a custom date helper for consistent formatting.
- Error messages are clear, user-friendly, and always in English.

### 4. **Task Management**
- The task list view allows users to filter, search, and manage tasks.
- Tasks can be filtered by status, type, and time window.
- The UI provides feedback for empty states, errors, and loading, with actionable buttons to clear filters or retry fetching.
- All tasks within a selected time window are fetched at once; there is currently no pagination.

### 5. **Training**
- Users can start new model training jobs, typically by selecting datasets, models, and parameters.
- The training module displays the status and progress of ongoing and past training jobs.
- Each training run has a detailed view, including logs, metrics, and results.
- Training jobs are managed and fetched using React Query, with robust error and loading state handling.
- Trained models can be directly evaluated or used elsewhere in the application.

### 6. **Upload**
- Users can upload new datasets or models through a dedicated upload interface.
- The upload process includes validation, progress feedback, and error handling.
- Uploaded resources are immediately available for use in training, evaluation, or other workflows.
- The upload feature is integrated with the rest of the app, ensuring a seamless user experience.

### 7. **UI/UX and Components**
- The UI uses Tailwind CSS for styling and utility classes.
- Components are organized by feature (e.g., `features/model`, `features/dataset`, `features/evaluation`, `features/training`, `features/upload`).
- Common UI elements (cards, buttons, tables, charts) are abstracted into reusable components under ui.
- The design is responsive, accessible, and visually consistent.

### 8. **Error Handling**
- All data fetching uses React Query, with loading and error states handled explicitly in the UI.
- If a model or baseline is missing, a non-clickable error card is shown with the message "Model not found".
- If an evaluation status fails to load, a clear error message is displayed.

### 9. **Code Quality and Best Practices**
- The codebase is TypeScript-first, ensuring type safety and maintainability.
- Linting and formatting are enforced (ESLint, Prettier).
- Functions are kept concise, and complex logic is extracted into helpers where possible.

---

## Example: Evaluation Detail View (`EvaluationData`)

- Fetches evaluation status using React Query.
- Renders a status card with progress, task ID, creation date, summary, and end date.
- Renders model and baseline cards, with existence checks and error handling.
- Renders dataset cards, or a message if no dataset is linked.
- Displays error messages in a user-friendly way.

---

## Project Structure

- app: Next.js app directory, with route-based folders for pages.
- features: Feature-based folders for models, datasets, evaluation, training, upload, tasks, etc.
- ui: Shared UI components (cards, buttons, charts, etc.).
- lib: Utility functions and API clients.
- public: Static assets (images, data files).
- resources: Configuration files (e.g., SonarQube).
- tests: End-to-end and integration tests.

---

## Technologies Used

- **Next.js**: Server-side rendering, routing, and API integration.
- **React**: Component-based UI.
- **TypeScript**: Type safety and better developer experience.
- **React Query**: Data fetching, caching, and synchronization.
- **Tailwind CSS**: Utility-first CSS framework for rapid UI development.
- **ESLint/Prettier**: Code quality and formatting.

---

## Extensibility and Maintenance

- The project is designed for easy extension: new features can be added as new folders under features.
- UI components are reusable and composable.
- Error handling and loading states are standardized across the app.
- The codebase is clean, idiomatic, and well-documented.
