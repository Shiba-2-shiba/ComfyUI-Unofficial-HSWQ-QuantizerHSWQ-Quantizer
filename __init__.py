import traceback

print("### HSWQ Nodes: Initializing... ###")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# 1. 統計コレクター (Calibration) の読み込み
try:
    from .SDXLQuantStatsCollector import NODE_CLASS_MAPPINGS as CalibMappings, NODE_DISPLAY_NAME_MAPPINGS as CalibDisplay
    NODE_CLASS_MAPPINGS.update(CalibMappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(CalibDisplay)
    print("  [OK] SDXLQuantStatsCollector loaded.")
except Exception as e:
    print("  [ERROR] Failed to import SDXLQuantStatsCollector")
    traceback.print_exc()

# 2. メイン量子化器 (V1.5 / Main) の読み込み
try:
    # ファイル名を SDXLHSWQQuantizer.py に変更した前提
    from .SDXLHSWQQuantizer import NODE_CLASS_MAPPINGS as QuantMappings, NODE_DISPLAY_NAME_MAPPINGS as QuantDisplay
    NODE_CLASS_MAPPINGS.update(QuantMappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(QuantDisplay)
    print("  [OK] SDXLHSWQQuantizer (V1.5) loaded.")
except Exception as e:
    print("  [ERROR] Failed to import SDXLHSWQQuantizer")
    traceback.print_exc()

# 3. レガシー量子化器 (V1.0) の読み込み (ファイルが存在する場合のみ)
try:
    # ファイル名を SDXLHSWQQuantizerLegacy.py に変更した前提
    from .SDXLHSWQQuantizerLegacy import NODE_CLASS_MAPPINGS as LegacyMappings, NODE_DISPLAY_NAME_MAPPINGS as LegacyDisplay
    NODE_CLASS_MAPPINGS.update(LegacyMappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(LegacyDisplay)
    print("  [OK] SDXLHSWQQuantizerLegacy (V1.0) loaded.")
except ImportError:
    pass # ファイルがない場合は無視
except Exception as e:
    print("  [ERROR] Failed to import SDXLHSWQQuantizerLegacy")
    traceback.print_exc()

# 4. ベンチマークツールの読み込み
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
