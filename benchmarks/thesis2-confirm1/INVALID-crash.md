# INVALID: interrupted by a host crash

The PC rebooted at ~21:44:45 on 2026-09-21 while the `amp-chain-gv` profile of
this campaign was running (14 of its 15 requests had completed; the server was
stopped by the crash, not by the harness).

Discard this campaign. Only `stock-mtp3` and `stock-be-mtp2` had their rows
flushed before the crash, and even those were recorded under the machine state
that preceded the crash. The results were not used in thesis.md.
