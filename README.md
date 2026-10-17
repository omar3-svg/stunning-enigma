# Team Bama — EcoCAR Autocross (HackBama 2026, Track 02)

A controller that drives an autocross course with **no map**: it only sees 40 m of road ahead and decides steering, throttle and brake every 50 ms.

## How it drives
1. **Racing line** – the straightest path inside the cones, with a safety margin (solved every step).
2. **Speed plan** – the fastest speed the tyres allow (friction circle), braking early enough for a sharp corner just past what it can see.
3. **Control** – pure pursuit steering, grip-limited pedals, and anti-spin.

## How we tested it
- **212 certified scenarios** (turn + hidden obstacle): our first version was clean on 138; after fixes, **209 / 212**.
- **9 stress conditions** (steering delay, low grip, steering bias, heavier car, noisy sensors…): **45 → 59 of 72** clean after adding delay compensation, bias correction and grip sensing.

## Results (official self-test, bicycle model)
| Course | Time | Cones | Points |
|---|---|---|---|
| hackathon_test | 47.8 s | 0 | 70.4 |
| practice_basic | 29.7 s | 0 | 72.9 |
| practice_loop | 82.1 s | 0 | 70.1 |
| practice_obstacles | 51.8 s | 0 | 70.7 |
| practice_technical | 35.5 s | 0 | 69.4 |
| **Mean** | | | **70.7 / 75** |

## Run it
Put `team_bama.py` in the simulator's `controllers` folder, then:
    python -m autocross selftest -c controllers/team_bama.py

Not yet tested on CARLA.
