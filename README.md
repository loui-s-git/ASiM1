# Franka Dashboard V18

Stand: 23.11.2026 · gilt für `Franka_Dashboard_V18.py`, `Franka_Robot_V18.py`, `drive_model.json` und `waypoints.json`

## Terminal-Befehle auf dem Linux-Rechner

Alle Befehle laufen im Terminal des Roboter-PCs (Ubuntu). Leg `Franka_Dashboard_V18.py`, `Franka_Robot_V18.py` und `drive_model.json` zusammen in denselben Ordner, hier `~/Downloads`. Gestartet wird nur `Franka_Dashboard_V18.py`, es lädt die beiden anderen Dateien selbst.

Starte das Dashboard immer aus diesem Ordner. `waypoints.json`, `dashboard_settings.json` und der Ordner `logs/` liegen relativ zum aktuellen Ordner.

Vor dem Start: Roboter in Desk entsperren und FCI aktivieren.

### Einmalig einrichten

```bash
source ~/franka_venv/bin/activate      # venv aus install_franka.sh (numpy, panda-python)
cd ~/Downloads
pip install -r requirements.txt        # installiert auch nicegui, das install_franka.sh nicht mitbringt
```

### Dashboard starten

```bash
source ~/franka_venv/bin/activate
cd ~/Downloads
python3 Franka_Dashboard_V18.py                # öffnet http://localhost:8080 im Browser
```

- Worker-Ansicht: http://localhost:8080/?view=worker. Dort bestätigt der Worker nach jedem Teil mit „Correct“ oder „Wrong“, ob die Entscheidung stimmt. Bei „Wrong“ wählt er das tatsächliche Teil. Die Eingabe landet als „Actual“ in Confusion Matrix, CSV und Summary.
- Die Roboter-IP stellst du im Dashboard unter „Connection“ ein (Standard 172.16.0.2), sie bleibt gespeichert. Die Wegpunktdatei steht unter „Data & files“.
- Beenden mit Strg+C im Terminal.

### Klassifikationsmodell trainieren

Das Modell lernt aus `drive_dimensions.csv`, welche Kombination aus Dicke (Fingeröffnung) und Gewicht zu SSD, HDD 2.5″ oder HDD 3.5″ gehört. Die CSV liest pandas ein (`read_csv`), trainiert wird mit scikit-learn: Train/Test-Split 75/25, logistische Regression, C per Kreuzvalidierung auf dem Trainingsteil. Das Skript gibt die Testgenauigkeit und die Confusion Matrix aus und schreibt `drive_model.json`. Das Dashboard selbst braucht weder pandas noch scikit-learn, es liest nur `drive_model.json`. Installier beides deshalb in einem eigenen venv, dann bleibt das Roboter-venv unverändert.

```bash
python3 -m venv ~/train_venv                   # einmalig
source ~/train_venv/bin/activate
pip install -r requirements-train.txt          # einmalig: pandas, scikit-learn, numpy
cd ~/Downloads
python3 train_drive_model.py                   # drive_dimensions.csv → drive_model.json
deactivate
```

Nach dem Training das Dashboard neu starten, es liest `drive_model.json` nur beim Start. Im Bereich „Classification“ schaltest du zwischen „Trained model“ und „Thresholds“ um.

## Teach-in-Ablauf

Alle 15 Pfade, jeder mit mindestens 5 Wegpunkten. Halte die Reihenfolge ein: Jeder Pfad muss dort beginnen, wo der vorige endet. Die Kästchen kannst du beim Teachen abhaken.

## Übersicht

| # | Pfad | Beginnt bei | Endet bei | Nötig |
|---|---|---|---|---|
| 1 | `to_holder` | Startpunkt | Greifpose an der Halterung | ja |
| 2 | `to_weigh` | Pose nach dem Hub | Wiegeplatz | ja |
| 3 | `start_to_weigh` | Startpunkt | Wiegeplatz | optional, empfohlen |
| 4 | `to_bench25` | Wiegeplatz | Anfahrpunkt über der 2.5″-Bench | ja |
| 5 | `out_bench25` | Sitz in der 2.5″-Bench | neutrale Pose neben den Benches | ja |
| 6 | `to_ssd` | Ende von `out_bench25` | Abwurf über der SSD-Kiste | ja |
| 7 | `home_ssd` | Abwurf über der SSD-Kiste | Startpunkt | ja |
| 8 | `to_hdd25` | Ende von `out_bench25` | Abwurf über der HDD-2.5″-Kiste | ja |
| 9 | `home_hdd25` | Abwurf über der HDD-2.5″-Kiste | Startpunkt | ja |
| 10 | `to_bench35` | Wiegeplatz | Anfahrpunkt über der 3.5″-Bench | ja |
| 11 | `out_bench35` | Sitz in der 3.5″-Bench | neutrale Pose neben den Benches | ja |
| 12 | `to_hdd35` | Ende von `out_bench35` | Abwurf über der HDD-3.5″-Kiste | ja |
| 13 | `home_hdd35` | Abwurf über der HDD-3.5″-Kiste | Startpunkt | ja |
| 14 | `to_reject` | Wiegeplatz | Abwurf über der Reject-Kiste | nur mit Reject |
| 15 | `home_reject` | Abwurf über der Reject-Kiste | Startpunkt | wenn `to_reject` geteacht |

## Vorbereitung und Grundregeln

- [ ] **Hubhöhe zuerst festlegen.** Stell „Lift height“ im Bereich Motion ein, zum Beispiel auf 50 mm. Danach änderst du sie nicht mehr, sonst passen `to_holder` und `to_weigh` nicht mehr zusammen.
- [ ] **Langsam anfangen.** Stell den „Speed factor“ für die ersten Testfahrten auf 0.1.
- [ ] **Greifer kalibrieren** mit „Home gripper“.
- [ ] **Laufwerke bereitlegen:** je ein SSD, eine HDD 2.5″ und eine HDD 3.5″.

Regeln für jeden Pfad:

- **Jeder Pfad beginnt mit dem Drive-Button.** Klick den Pfad in der Liste an und drück „Drive to end of previous path“. Danach drückst du sofort W, ohne den Arm zu bewegen. Steht der Arm schon am Ende des vorigen Pfads, fährt der Button nicht los.
- **Öffne den Greifer vor jeder Drive-Fahrt** mit „Open gripper“. Die Fahrt führt meist über die Greifpose an der Halterung. Ein geschlossener Greifer würde dort auf ein Laufwerk stoßen.
- **Fahr senkrechte Stücke nie von Hand.** Nimm dafür die Straight-Buttons mit 10 mm und 2 mm. Sie halten x, y und die Drehung exakt.
- **Der Arm fährt zwischen zwei Punkten keine gerade Linie.** Setz die Punkte deshalb dicht an Halterung, Bench und Kisten. Im freien Raum reichen größere Abstände.
- **Klapp die Gelenkstellung nicht um.** Lass den Ellbogen oben und dreh das Handgelenk zwischen zwei Punkten nur wenig. In der Telemetrie sollte kein Gelenk gelb werden.
- **Zwischen zwei W muss sich der Arm bewegen.** Sonst verwirft das Dashboard den Punkt.
- **Lies nach jedem Speichern das Log.** Dort darf keine Meldung „Gap after saving“ stehen. Bei den Bench-Pfaden muss „Bench seat OK“ erscheinen.
- **Teach mit echtem Laufwerk, wo es um Abstände geht.** Halt das Laufwerk zwischen die Finger und drück „Test grasp“. Der Greifer schließt dann, ohne dass der Arm fährt.
- **Eine 3.5″-HDD zieht den Arm im Teaching-Modus etwas nach unten,** weil ihr Gewicht nicht kompensiert wird. Stütz den Arm dann mit der Hand.
- Beim ersten Speichern einer Sitzung legt das Dashboard automatisch `waypoints.json.bak` an.

## 1. `to_holder`: vom Startpunkt zur Greifpose

Stell ein Laufwerk in die Halterung und öffne den Greifer.

- [ ] **P1 Startpunkt:** frei über dem Tisch, Greifer zeigt nach unten. Alle Wege zurück enden hier. Willst du den alten Startpunkt behalten, fährst du vorher „Go to start“ und drückst dort W.
- [ ] **P2 Zwischenpunkt** im freien Raum in Richtung Halterung.
- [ ] **P3 Zwischenpunkt** 10 bis 15 cm über der Halterung, Greifer schon ausgerichtet.
- [ ] **P4 genau über der Greifpose, in Hubhöhe:** Führ den offenen Greifer von Hand in die Greifpose und drück noch kein W. Fahr dann mit den Straight-Buttons um die Hubhöhe hoch, bei 50 mm also fünfmal 10 mm. Erst dann W.
- [ ] **P5 Greifpose:** Fahr mit den Straight-Buttons dieselbe Strecke wieder runter und drück W.
- [ ] **Speichern.**

P4 ist auch der Rückweg nach einem leeren Griff und nach dem Zurücklegen. Deshalb muss er exakt über der Greifpose liegen.

## 2. `to_weigh`: von der Pose nach dem Hub zum Wiegeplatz

Der Drive-Button fährt `to_holder`, hebt an und hält genau in P1.

- [ ] **P1 Pose nach dem Hub:** sofort W.
- [ ] **P2 weiter hoch:** Mit den Straight-Buttons hoch, bis ein Laufwerk ganz aus der Halterung wäre und noch 2 cm Luft hätte. W.
- [ ] **P3 Zwischenpunkt** in Richtung Wiegeplatz.
- [ ] **P4 Zwischenpunkt** kurz vor dem Wiegeplatz.
- [ ] **P5 Wiegeplatz:** frei in der Luft, nichts berührt Hand oder Kabel. Hier wird jedes Teil tariert und gewogen. W.
- [ ] **Speichern.**

## 3. `start_to_weigh`: leer vom Startpunkt zum Wiegeplatz

Optional, aber empfohlen. Ohne diesen Pfad springt der leere Arm direkt vom Startpunkt über die Halterung. Der Drive-Button fährt zum Startpunkt.

- [ ] **P1 Startpunkt:** sofort W.
- [ ] **P2 Zwischenpunkt** auf einem Weg, der die Halterung meidet.
- [ ] **P3 Zwischenpunkt.**
- [ ] **P4 Zwischenpunkt** nahe am Wiegeplatz.
- [ ] **P5 Wiegeplatz:** „Finish at the weigh spot“ drücken. Damit wird der exakte Wiegeplatz als letzter Punkt angehängt.
- [ ] **Speichern.**

## 4. `to_bench25`: vom Wiegeplatz zum Anfahrpunkt über der 2.5″-Bench

Nach dem Drive-Button steht der Arm am Wiegeplatz. Nimm dort ein SSD mit „Test grasp“ in den Greifer.

- [ ] **P1 Wiegeplatz:** sofort W.
- [ ] **P2 Zwischenpunkt** in Richtung Bench.
- [ ] **P3 Zwischenpunkt** mit Abstand zur 3.5″-Bench. Die beiden Benches stehen nur 68 mm auseinander.
- [ ] **P4 Zwischenpunkt** 10 bis 15 cm über der Bench, Greifer schon so gedreht wie beim Einsetzen.
- [ ] **P5 Anfahrpunkt:** Führ das SSD von Hand in den Slot, bis es sitzt, und drück noch kein W. Fahr dann mit den Straight-Buttons hoch, bis die Unterkante des SSD 2 bis 5 cm über der Bench ist. **Zähl dabei die Schritte.** W.
- [ ] **Speichern** und den Arm nicht mehr bewegen.

## 5. `out_bench25`: vom Sitz in der 2.5″-Bench bis frei darüber

Direkt im Anschluss, der Arm steht noch am Anfahrpunkt und hält das SSD.

- [ ] **P1 Sitz:** Fahr mit den Straight-Buttons genau dieselben Schritte wieder runter, die letzten mit 2 mm. Das SSD sitzt dann wieder ganz in der Bench. W.
- [ ] **P2:** 10 mm hoch. W.
- [ ] **P3:** weiter hoch, bis das SSD ganz aus der Bench ist und 2 cm Luft hat. W.
- [ ] **P4:** von der Bench weg in Richtung Kisten, auf sicherer Höhe. W.
- [ ] **P5 neutrale Pose,** frei von beiden Benches. Hier starten `to_ssd` und `to_hdd25`. W.
- [ ] **Speichern.** Im Log muss „Bench seat OK“ stehen.

## 6. `to_ssd`: vom Ende von `out_bench25` bis über die SSD-Kiste

Direkt im Anschluss, das SSD ist noch im Greifer.

- [ ] **P1 Ende von `out_bench25`:** sofort W.
- [ ] **P2 Zwischenpunkt** um Hindernisse herum.
- [ ] **P3 Zwischenpunkt** um Hindernisse herum.
- [ ] **P4:** 10 bis 15 cm über der SSD-Kiste.
- [ ] **P5 Abwurfpose:** mittig über der Kiste, die Unterkante des SSD wenige Zentimeter über dem Rand. W.
- [ ] **Speichern.** Drück dann „Open gripper“, damit das SSD in die Kiste fällt.

## 7. `home_ssd`: von der SSD-Kiste zurück zum Startpunkt

Direkt im Anschluss.

- [ ] **P1 Abwurfpose:** sofort W.
- [ ] **P2:** mit den Straight-Buttons senkrecht hoch, aus der Kiste heraus. W.
- [ ] **P3 Zwischenpunkt** um Hindernisse herum.
- [ ] **P4 Zwischenpunkt** nahe am Startpunkt.
- [ ] **P5 Startpunkt:** „Finish at the start point“ drücken. Damit wird der exakte Startpunkt angehängt.
- [ ] **Speichern.**

Lässt du P5 weg, hängt das Programm den Startpunkt trotzdem an. Es fährt dann aber einen ungeteachten Sprung, im letzten Log bis zu 1.67 rad.

## 8. `to_hdd25`: vom Ende von `out_bench25` bis über die HDD-2.5″-Kiste

Der Drive-Button fährt die ganze Kette bis zum Ende von `out_bench25`. Nimm dort eine HDD 2.5″ mit „Test grasp“ in den Greifer.

- [ ] **P1 Ende von `out_bench25`:** sofort W.
- [ ] **P2 Zwischenpunkt** um Hindernisse herum.
- [ ] **P3 Zwischenpunkt** um Hindernisse herum.
- [ ] **P4:** 10 bis 15 cm über der HDD-2.5″-Kiste.
- [ ] **P5 Abwurfpose:** mittig über der Kiste, die Unterkante der HDD wenige Zentimeter über dem Rand. W.
- [ ] **Speichern.** Drück dann „Open gripper“.

## 9. `home_hdd25`: von der HDD-2.5″-Kiste zurück zum Startpunkt

Direkt im Anschluss.

- [ ] **P1 Abwurfpose:** sofort W.
- [ ] **P2:** mit den Straight-Buttons senkrecht hoch, aus der Kiste heraus. W.
- [ ] **P3 Zwischenpunkt** um Hindernisse herum.
- [ ] **P4 Zwischenpunkt** nahe am Startpunkt.
- [ ] **P5 Startpunkt:** „Finish at the start point“ drücken.
- [ ] **Speichern.**

## 10. `to_bench35`: vom Wiegeplatz zum Anfahrpunkt über der 3.5″-Bench

Nach dem Drive-Button steht der Arm am Wiegeplatz. Nimm dort eine HDD 3.5″ mit „Test grasp“ in den Greifer und stütz den Arm.

- [ ] **P1 Wiegeplatz:** sofort W.
- [ ] **P2 Zwischenpunkt** in Richtung Bench.
- [ ] **P3 Zwischenpunkt** mit Abstand zur 2.5″-Bench.
- [ ] **P4 Zwischenpunkt** 10 bis 15 cm über der Bench, Greifer schon so gedreht wie beim Einsetzen.
- [ ] **P5 Anfahrpunkt:** Führ die HDD von Hand in den Slot, bis sie sitzt, und drück noch kein W. Fahr dann mit den Straight-Buttons hoch, bis die Unterkante 2 bis 5 cm über der Bench ist. **Zähl dabei die Schritte.** W.
- [ ] **Speichern** und den Arm nicht mehr bewegen.

## 11. `out_bench35`: vom Sitz in der 3.5″-Bench bis frei darüber

Direkt im Anschluss, der Arm steht noch am Anfahrpunkt und hält die HDD.

- [ ] **P1 Sitz:** Fahr mit den Straight-Buttons genau dieselben Schritte wieder runter, die letzten mit 2 mm. W.
- [ ] **P2:** 10 mm hoch. W.
- [ ] **P3:** weiter hoch, bis die HDD ganz aus der Bench ist und 2 cm Luft hat. W.
- [ ] **P4:** von der Bench weg in Richtung Kisten, auf sicherer Höhe. W.
- [ ] **P5 neutrale Pose,** frei von beiden Benches. Hier startet `to_hdd35`. W.
- [ ] **Speichern.** Im Log muss „Bench seat OK“ stehen.

## 12. `to_hdd35`: vom Ende von `out_bench35` bis über die HDD-3.5″-Kiste

Direkt im Anschluss, die HDD ist noch im Greifer.

- [ ] **P1 Ende von `out_bench35`:** sofort W.
- [ ] **P2 Zwischenpunkt** um Hindernisse herum.
- [ ] **P3 Zwischenpunkt** um Hindernisse herum.
- [ ] **P4:** 10 bis 15 cm über der HDD-3.5″-Kiste.
- [ ] **P5 Abwurfpose:** mittig über der Kiste, die Unterkante der HDD wenige Zentimeter über dem Rand. W.
- [ ] **Speichern.** Drück dann „Open gripper“.

## 13. `home_hdd35`: von der HDD-3.5″-Kiste zurück zum Startpunkt

Direkt im Anschluss.

- [ ] **P1 Abwurfpose:** sofort W.
- [ ] **P2:** mit den Straight-Buttons senkrecht hoch, aus der Kiste heraus. W.
- [ ] **P3 Zwischenpunkt** um Hindernisse herum.
- [ ] **P4 Zwischenpunkt** nahe am Startpunkt.
- [ ] **P5 Startpunkt:** „Finish at the start point“ drücken.
- [ ] **Speichern.**

## 14. `to_reject`: vom Wiegeplatz bis über die Reject-Kiste

Nur nötig, wenn „Reject implausible parts“ an ist. Rejects fahren nicht über die Bench. Nach dem Drive-Button steht der Arm am Wiegeplatz. Nimm dort ein beliebiges Laufwerk mit „Test grasp“ in den Greifer.

- [ ] **P1 Wiegeplatz:** sofort W.
- [ ] **P2 Zwischenpunkt** um Hindernisse herum.
- [ ] **P3 Zwischenpunkt** um Hindernisse herum.
- [ ] **P4:** 10 bis 15 cm über der Reject-Kiste.
- [ ] **P5 Abwurfpose:** mittig über der Kiste, die Unterkante des Laufwerks wenige Zentimeter über dem Rand. W.
- [ ] **Speichern.** Drück dann „Open gripper“.

## 15. `home_reject`: von der Reject-Kiste zurück zum Startpunkt

Pflicht, sobald `to_reject` geteacht ist. Direkt im Anschluss.

- [ ] **P1 Abwurfpose:** sofort W.
- [ ] **P2:** mit den Straight-Buttons senkrecht hoch, aus der Kiste heraus. W.
- [ ] **P3 Zwischenpunkt** um Hindernisse herum.
- [ ] **P4 Zwischenpunkt** nahe am Startpunkt.
- [ ] **P5 Startpunkt:** „Finish at the start point“ drücken.
- [ ] **Speichern.**

## Prüfen und Testen

- [ ] **Statuschips grün.** Im Sortierzyklus müssen vor allem „Paths 12/12“, „Paths connected“ und „Bench seats“ grün sein.
- [ ] **Test drive je Route.** Fahr nacheinander die Routen SSD, HDD 2.5″, HDD 3.5″ und Reject, mit der Hand am STOP. Der Bestätigungsdialog listet ungeteachte Direktfahrten auf. Erwartet ist dort nur der kurze Schritt vom Anfahrpunkt hinunter zum Sitz in der Bench.
- [ ] **Dry Run pro Kiste,** ohne Teil.
- [ ] **„Measure only“** mit je einem echten Teil pro Typ. Das prüft Greifen, Hub, Wiegen und Zurücklegen.
- [ ] **„Sort part“ einzeln,** je ein Teil pro Typ.
- [ ] **Auto-Modus.**
- [ ] **Geschwindigkeit schrittweise erhöhen.**
