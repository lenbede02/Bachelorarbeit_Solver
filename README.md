# Solver Code der Bachelorarbeit von Lennart Bederke. Für Ralf Klessen und Cornelis Dullemond

In diesem Repo finden Sie den Code den ich genutzt habe um die Ergebnisse meiner Bachelorarbeit zu generieren. Der Hauptkern ist dabei der Solver: disk_v3_erweitert_linblad.py. Die restlichen Dateien nutzen ihn um die Ergebnisse der BA zu generieren. Falls Sie eigene Testläufe machen wollen, empfehle ich die bereits vorhandene "Simulate" Funktion in dieser Hauptdatei zu nutzen, da hier die ICs und Parameter einfach geändert werden können. Sie finden die Funktion ganz am Ende der Datei. Beachten Sie, dass Sie den Schieberegler ausstellen können, falls Sie nicht an der Zeitentwicklung interessiert sind. Das geht durch das Argument "interactive = True / False". Falls Sie Fragen haben oder irgendwas nicht klappt, können Sie mich gerne jederzeit kontaktieren. 


## Code

| Datei | Rolle |
|---|---|
| `disk_v3_erweitert_linblad.py` | Der eigentliche Solver 
| `verifikation.py` | Tests V0–V12: Abbildungen und Kennzahlentabelle. |
| `parameterstudie_linblad.py` | Referenzlauf, vier 1D-Sweeps, zwei 2D-Karten, Multi-Populations-Lauf. |
| `disk_v3_erweitert.py` | Vorgängerstand ohne Planetenterm. Wird nur von Test V0 gebraucht, der zeigt, dass der erweiterte Solver bei q=0 exakt in ihn übergeht. Ein Artefakt des Entwicklungsprozesses, dass ich in den Verifikationsläufen belassen habe. Damit der Code ausgeführt werden kann bleibt er hier erhalten. |
| `numval.py` | Protokolliert zu jeder gespeicherten Abbildung die geplotteten Rohwerte als CSV nach `numVal/`. Diese Datei enthält keine Physik sondern ist primär ein Helper um nach einem run noch auf die Zahlen des letzten runs zugreifen zu können|
| `progress.py` | Fortschrittsbalken. |
| `druckprofil_figure.py`, `peclet_collapse_figure.py`, `rwi_figure.py`, `backreaction_compare.py` | je eine Abbildung  |


## Reproduktion

Alle Aufrufe aus `solver/` heraus. Die PDFs landen dann in
`thesis/figures/…`. Die Lauzeit betrug bei mir etwa 10 Stunden. Die eigentliche Zeit kann sich natürlich von Gerät zu Gerät unterscheiden. Da die Rechenzeit im allgemeinen sehr lang ist empfehle ich die nicht benötigten bzw. gewollten Teile der Parameterstudie oder der Verifikation einfach vorher kurz auszukommentieren. Vor allem die Karten sind sehr rechenintensiv.

`peclet_collapse_figure.py` benötigt, dass zuerst die Parameterstudie gelaufen ist, da die Datei mit den Werten aus der CSV arbeitet.

## CSV

Jeder Lauf legt neben den Abbildungen die tatsächlich geplotteten Werte als CSV
unter `solver/numVal/` ab. Damit lässt sich jede Kurve und jede im Text
zitierte Zahl direkt gegen die Daten prüfen. Die Kennzahlen aller Verifikationstests stehen gesammelt in `numVal/verifikation/V0-V11_zusammenfassung.csv`.
