# PixieNN model graph site

Install the model graph's own frontend dependencies, then build the React frontend:

```bash
cd model-graph
npm install
npm run build
```

Serve it with the Python YAML API:

```bash
python3 model-graph/server.py --model resources/models/centernet-prosopo.yml
```

Then open <http://127.0.0.1:5179/>.  Use **Load YML** to inspect another
PixieNN model without restarting the server.
