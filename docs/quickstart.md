# Quickstart

1. Install
   - CPU: `pip install -r requirements.txt`
   - CUDA: `pip install -r requirements-cuda.txt`

2. Run HTTP
   - Press F5 and choose "SATOR: Run Server" (recommended), or
   - Set your key in `.env` as `SATOR_API_KEY=dev-key`, then run `sator-server`
   - For HTTPS (TLS) locally, see `local_tls_certs.md` and ensure TLS vars are set in `.env`.

3. Optimize
   - POST /v1/optimize with JSON body including dataset, objectives, search_space and optimization_config
   - Example (minimal):
     ```bash
     curl -s -H "x-api-key: dev-key" -H "Content-Type: application/json" \
       -d '{"dataset":{},"search_space":{"parameters":[{"name":"x1","type":"float","min":0,"max":1}]},"objectives":{"o1":{"goal":"min"}},"optimization_config":{"acquisition":"qnehvi","batch_size":2,"max_evaluations":10}}' \
       http://localhost:8080/v1/optimize
     ```

   - Or run the examples (recommended):
     - `python .\\examples\\http_visualize_branin.py` (3D surface)
     - `python .\\examples\\http_chem_pca_optimize_visualize.py` (PCA(2) maps, blue=training, red=predictions)

4. Poll for result
   - GET /v1/jobs/{job_id}/result

