#!/bin/sh

SCRIPT_DIR=$(
  cd $(dirname $0)
  cd ../
  pwd
)

. ${SCRIPT_DIR}/.venv/bin/activate

while true; do
  python ${SCRIPT_DIR}/insert_future_ohlcv.py
done
