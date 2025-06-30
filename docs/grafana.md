
## Monitoring with Grafana

Vectorize comes with **powerful built-in observability** through Grafana, Prometheus, and Loki, giving you full visibility into your system's behavior, performance, and health.


### Accessing Grafana

- **URL:** [https://localhost/monitoring](https://localhost/monitoring)
- **Login Credentials:**
  - **Username:** `admin`
  - **Password:** `admin`  
    _(You can update these credentials in `resources/grafana/compose.yaml` and related config files.)_

---


### What You’ll See

- **Custom Dashboard:**  
  Find a pre-configured dashboard under **Dashboards**, designed specifically for Vectorize. It shows:

  - Request rates
  - Error rates
  - Latency trends
  - Background task status

- **Deep Dive with Explore:**
  - **Explore > Prometheus:** Browse real-time metrics, like HTTP request rates, response times, and custom counters.
  - **Explore > Loki:** Search and analyze logs with structured fields and full-text search.
  - **Explore > Alloy:** View processed logs and metrics for advanced insights.

---


### How It Works

- **Prometheus Metrics:**  
  Vectorize exposes detailed metrics via a custom middleware in `src/vectorize/utils/prometheus.py`.  
  Prometheus scrapes these metrics on a schedule, storing them for analysis in Grafana.

- **Loki Logs:**  
  Application logs are structured and forwarded to Loki, configured through `src/vectorize/config/logger.py`.  
  Loki provides fast, indexed searches across logs, letting you correlate errors and performance events.

- **Alloy Processing:**  
  Alloy handles advanced log processing and routing, ensuring logs are enriched before reaching Loki.

---


### Customization Options

- **Update Admin Credentials:**  
  Set your own Grafana admin username and password by editing environment variables in `resources/grafana/compose.yaml`.

- **Tailor Dashboards:**  
  Modify or extend the default dashboard directly in Grafana, or import your own JSON dashboards for custom views.

- **Tweak Prometheus & Loki:**
  - **Prometheus scraping:** `resources/grafana/config/prometheus.yaml`
  - **Loki pipeline:** `resources/grafana/config/loki-config.yaml`
  - **Alloy config:** `resources/grafana/config/config.alloy`

---


### Useful Tips

- **Live Monitoring:**  
  Keep the Grafana dashboard open for real-time tracking of API performance, background task activity, and error rates.
- **Troubleshooting Made Easy:**  
  Use the **Explore** tab to jump between metrics and logs, making it easier to identify the root cause of issues.
- **Secure Your Instance:**  
  Always update default admin credentials before deploying to production to keep your monitoring stack secure.

---

Grafana gives you a **clear window into your Vectorize deployment**, with the ability to customize every aspect of your monitoring. For deeper details, check out the configuration files in `resources/grafana/`.
