# Dissertation Project Audit and Improvement Plan

## Key issues found in the uploaded codebase

1. The live dashboard said “live” but always used `PaperTradingEngine`, so it never routed orders to MT5.
2. `MT5Connector` could connect and read ticks, but had no trade-send path.
3. The dashboard learned from unresolved predictions immediately, which created self-reinforcing bias.
4. Live accuracy was effectively fake because outcomes were never resolved before `record_outcome()`.
5. The confidence head in the model was never trained, so the old confidence thresholds were not meaningful.
6. The old saved `.twin_state` came from the biased logic and carried the same bad behaviour forward.
7. Twin-B drifted into one-sided BUY behaviour because replay labels were being polluted by its own predictions.
8. Twin-A rarely acted because entropy/confidence gates were too strict relative to the actual outputs.
9. Spread filtering existed in the scanner but did not actually block execution.
10. Gold symbol handling assumed exact `XAUUSD`, which breaks on brokers with suffixes or `GOLD` aliases.

## What has been implemented now

1. Added real MT5 order routing capability.
2. Added explicit execution-mode selection: `Paper` or `Live MT5`.
3. Added a required live-order enable switch before real orders can be sent.
4. Added trade-permission checks from both account and terminal.
5. Added broker-symbol auto-discovery for `XAUUSD` / `GOLD` variants.
6. Added an MT5 execution engine that mirrors the paper-engine analytics interface.
7. Added broker ticket tracking inside live order records.
8. Added live open-position syncing from MT5.
9. Added broker exit syncing from MT5 deal history.
10. Fixed live learning so twins only learn from resolved outcomes.
11. Added actual-label propagation through `OutcomeEvaluator`.
12. Removed self-labelling bias from replay updates.
13. Added horizon-aligned pretraining labels.
14. Added class balancing in the pretraining dataset.
15. Added hybrid direction logic to stabilise model bias.
16. Reworked runtime confidence to be calibrated from directional edge and uncertainty.
17. Tuned Twin-A and Twin-B gating so both can act.
18. Activated the spread filter as a real entry blocker.
19. Added bias visibility in the live dashboard.
20. Added state versioning so stale/bad saved states are ignored.
21. Generated a new compatible saved twin state.
22. Added a smoke test script.
23. Updated requirements to include optional MetaTrader5 support on Windows.
24. Removed misleading “paper only” / fake-live wording from the dashboard path.

## 50 high-grade functionality improvements to consider

### Already implemented in this delivery

1. Real MT5 execution path.
2. Paper/live execution switch.
3. Live-order safety enable toggle.
4. Terminal/account trade permission diagnostics.
5. Gold symbol auto-discovery.
6. MT5 position sync.
7. MT5 exit-deal sync.
8. Broker ticket visibility.
9. Outcome-horizon learning fix.
10. Actual-label replay learning.
11. Removal of self-training bias loop.
12. Balanced pretraining labels.
13. Horizon-aligned supervision.
14. Confidence recalibration.
15. Hybrid bias-corrected direction inference.
16. Activated spread blocker.
17. Bias monitoring card.
18. Saved-state versioning.
19. New compatible saved weights.
20. Smoke test coverage.

### Strong next upgrades

21. Dynamic position sizing from account risk % instead of fixed lots.
22. ATR-based stop loss and take profit instead of fixed pip distances.
23. Time-of-day session filters (Asia / London / New York).
24. News blackout windows around high-impact macro releases.
25. Broker-side trailing stop support.
26. Break-even stop logic after partial move in favour.
27. Partial take-profit laddering.
28. Separate per-twin max daily loss limits.
29. Global portfolio max drawdown lockout.
30. Daily trade count limiter.
31. Consecutive-loss circuit breaker at portfolio level.
32. Slippage logging for real MT5 fills.
33. Requote / reject retry policy with bounded attempts.
34. CSV / Parquet audit trail for every broker request and response.
35. Database-backed trade and signal storage.
36. Walk-forward validation pipeline for synthetic/live replay.
37. Calibration curves and Brier score tracking.
38. SHAP / feature attribution snapshot for each decision.
39. Ensemble of different model seeds per twin.
40. Optuna or Bayesian tuning for thresholds and retrain cadence.
41. Multi-symbol support beyond gold.
42. Telegram / email alerts on new trades and retrains.
43. Webhook integration for external monitoring.
44. Prometheus-style runtime metrics export.
45. Docker packaging for repeatable deployment.
46. CI pipeline running smoke tests on each change.
47. Secrets loading from environment variables instead of UI entry only.
48. Config file for all risk/execution parameters.
49. Equity-based adaptive throttling of both twins.
50. Live/paper divergence dashboard showing what would have happened in both modes.

## Files changed

- `live_dashboard.py`
- `models.py`
- `feedback.py`
- `utils.py`
- `twin_state.py`
- `mt5_bridge/mt5_connector.py`
- `mt5_bridge/mt5_trade_engine.py` *(new)*
- `mt5_bridge/__init__.py`
- `app.py`
- `requirements.txt`
- `smoke_test.py` *(new)*

## Quick run notes

### Training app
```bash
streamlit run app.py
```

### Live dashboard
```bash
streamlit run live_dashboard.py
```

### Smoke test
```bash
python smoke_test.py
```

## Important live-trading note

The code now supports real MT5 execution, but the user must deliberately choose `Live MT5`, connect successfully, and tick the real-order enable box before any broker order can be sent.
