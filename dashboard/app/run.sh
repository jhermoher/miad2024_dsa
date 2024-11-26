#!/bin/bash
export PORT=${PORT:-8050}
gunicorn app:server -b :${PORT} --timeout 600