import traceback

print("### HSWQ Nodes: Initializing... ###")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# ============================================================
# SDXL Series (V1.5 / Legacy)
# ============================================================

# 1. SDXL 統計コレクター (Calibration) の読み込み
try:
    from .SDXLQuantStatsCollector import NODE_CLASS_MAPPINGS as CalibMappings, NODE_DISPLAY_NAME_MAPPINGS as CalibDisplay
    NODE_CLASS_MAPPINGS.update(CalibMappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(CalibDisplay)
    print("  [OK] SDXLQuantStatsCollector loaded.")
except Exception as e:
    print("  [ERROR] Failed to import SDXLQuantStatsCollector")
    traceback.print_exc()

# 2. SDXL メイン量子化器 (V1.5 / Main) の読み込み
try:
    from .SDXLHSWQQuantizer import NODE_CLASS_MAPPINGS as QuantMappings, NODE_DISPLAY_NAME_MAPPINGS as QuantDisplay
    NODE_CLASS_MAPPINGS.update(QuantMappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(QuantDisplay)
    print("  [OK] SDXLHSWQQuantizer (V1.5) loaded.")
except Exception as e:
    print("  [ERROR] Failed to import SDXLHSWQQuantizer")
    traceback.print_exc()

# 3. SDXL レガシー量子化器 (V1.0) の読み込み
try:
    from .SDXLHSWQQuantizerLegacy import NODE_CLASS_MAPPINGS as LegacyMappings, NODE_DISPLAY_NAME_MAPPINGS as LegacyDisplay
    NODE_CLASS_MAPPINGS.update(LegacyMappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(LegacyDisplay)
    print("  [OK] SDXLHSWQQuantizerLegacy (V1.0) loaded.")
except ImportError:
    pass 
except Exception as e:
    print("  [ERROR] Failed to import SDXLHSWQQuantizerLegacy")
    traceback.print_exc()

# ============================================================
# ZIT Series (New)
# ============================================================

# 4. ZIT 統計コレクター (Calibration) の読み込み
try:
    from .ZITQuantStatsCollector import NODE_CLASS_MAPPINGS as ZITCalibMappings, NODE_DISPLAY_NAME_MAPPINGS as ZITCalibDisplay
    NODE_CLASS_MAPPINGS.update(ZITCalibMappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(ZITCalibDisplay)
    print("  [OK] ZITQuantStatsCollector loaded.")
except Exception as e:
    print("  [ERROR] Failed to import ZITQuantStatsCollector")
    traceback.print_exc()

# 5. ZIT 量子化器 (Spec-aligned) の読み込み
try:
    from .ZITHSWQQuantizer import NODE_CLASS_MAPPINGS as ZITQuantMappings, NODE_DISPLAY_NAME_MAPPINGS as ZITQuantDisplay
    NODE_CLASS_MAPPINGS.update(ZITQuantMappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(ZITQuantDisplay)
    print("  [OK] ZITHSWQQuantizer loaded.")
except Exception as e:
    print("  [ERROR] Failed to import ZITHSWQQuantizer")
    traceback.print_exc()

# ============================================================
# Tools / Benchmarks
# ============================================================

# 6. ベンチマークツールの読み込み
try:
    from .HSWQAdvancedBenchmark import NODE_CLASS_MAPPINGS as BenchMappings, NODE_DISPLAY_NAME_MAPPINGS as BenchDisplay
    NODE_CLASS_MAPPINGS.update(BenchMappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(BenchDisplay)
    print("  [OK] HSWQAdvancedBenchmark loaded.")
except Exception as e:
    print("  [ERROR] Failed to import HSWQAdvancedBenchmark")
    traceback.print_exc()

print(f"### HSWQ Nodes: Initialization complete. Total nodes: {len(NODE_CLASS_MAPPINGS)} ###")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
