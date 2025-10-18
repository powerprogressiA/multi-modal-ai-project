#!/usr/bin/env python3
"""
BrainFlow template for supported EEG boards. Check BrainFlow docs for Emotiv specifics.
"""
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import time

params = BrainFlowInputParams()
params.serial_port = '/dev/ttyUSB0'
board_id = 0  # replace with correct BoardIds.* if supported

BoardShim.enable_dev_board_logger()
board = BoardShim(board_id, params)
try:
    board.prepare_session()
    board.start_stream()
    time.sleep(2)
    data = board.get_board_data()
    print("Data shape:", data.shape)
finally:
    try:
        board.stop_stream()
        board.release_session()
    except Exception:
        pass
