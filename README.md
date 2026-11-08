# Encore

**His 9 AM voice, on cue at 4 PM.**

BioHacks 2026, Track 3: Wearable Devices and Quality of Life.

Encore lets a person with ALS keep speaking in their own voice as their speech
and movement change. They record the phrases that matter to them while their
voice is strongest, then give each phrase a personal cue: a short spoken word
(slurred is fine, as long as it is *their* slur) or a small face or head
movement. When Encore detects the cue, it plays the clear recording and shows
the words on screen.

One matching engine serves every kind of cue. When a cue gets hard, the user
re-records it with whatever still works, in about 30 seconds. The device adapts
to the person, not the other way round.

## Why it is built this way (from Alex's file)

| Finding in the case | Design decision |
|---|---|
| Speech 165 wpm / 98% intelligible at 9 AM, 118 wpm / 91% at 4 PM | Phrases are recorded in the morning and replayed in the afternoon. The recorder nudges morning sessions. |
| Students at the back ask him to repeat after lunch; he dropped the end-of-day recap | A classroom deck, including "Here's today's recap", which he can re-record each morning. Every phrase is also captioned on screen. |
| Dysarthric speech defeats off-the-shelf speech recognition | Cues are matched against his own examples (few-shot DTW), not a generic model. Slower speech is absorbed by time-warping. |
| Right pinch lasts seconds, left hand is starting to weaken | Gesture cues use the face and head, which outlast the hands. Eye movements are spared in ALS. |
| Reduced verbal fluency and executive scores (ECAS) | No menus to navigate. Small decks of ready-made phrases also help with word-finding. |
| Alone at home until 4:30, two falls after 6 PM | A home deck can send check-in, call-me and help notifications to Jordan. |
| Sunday cooking is what he least wants to give up | A kitchen deck so he can direct the cooking with his hands and voice tired. |
| Avoids eating in the staff room; meals take 35-40 min | "I'm taking my time with this. Keep talking, I'm listening." |
| Voice banking "feels like giving up" | Encore is framed as a teaching tool. The phrases he records for class are also, with his consent, clean voice data for a clone later. |
| ALS clinic reviews him with a questionnaire every 3 months | Every cue attempt is logged, giving an hour-by-hour picture of his voice he can choose to share. |

## Quick start

```bash
pip install numpy scipy            # enough for the demo and tests
python -m encore demo              # end-to-end demo on synthetic signals, no mic
python -m unittest discover -s tests -v
```

For live use install everything in `requirements.txt`.

## Using it

```bash
python -m encore init --seed            # phrase bank + starter decks from the case
python -m encore phrases                # list phrases with their ids
python -m encore record-phrase 7        # record phrase 7 in your own voice (mornings)
python -m encore record-phrase 7 --from-file old_clip.wav   # or import an old recording
python -m encore train-voice 7          # teach a spoken cue (3 examples)
python -m encore train-gesture 7        # or a face/head gesture cue (webcam)
python -m encore train-gesture arm      # arm cue for classroom mode (pick a gesture you
                                        # don't make while talking, e.g. a long blink)
python -m encore run --gesture          # listen to mic + camera
python -m encore run --require-arm      # classroom mode: cues only fire after the arm cue
python -m encore report --days 7        # hour-by-hour voice cue summary
python -m encore notify-setup <long-random-topic> --name Alex --test
python -m encore train-gesture 7 --replace   # re-record a cue that got hard
```

While `run` is active, typing a phrase number and Enter speaks it directly
(a keyboard or single-switch fallback).

## How it works

```
mic --> AudioVAD --> MFCC --+
                            +--> TemplateMatcher (DTW, per-cue thresholds) --> Engine
webcam --> FaceMesh --> baseline-normalised features --> GestureSegmenter --+       |
                                                                                    v
                         play his recording + caption + optional notification <- PhraseBank (SQLite)
                                                                                    |
                         AdaptationMonitor (voice / hybrid / gesture advice) <------+
```

| Module | Role |
|---|---|
| `db.py` | SQLite phrase bank: phrases, cue templates (old ones retired, not deleted), events, settings |
| `features.py` | MFCC + deltas in numpy, per-utterance normalisation |
| `matcher.py` | Banded DTW; each cue's threshold is calibrated from the spread of the user's own examples; ambiguity margin between cues |
| `segmenter.py` | One streaming state machine for audio (adaptive noise floor) and gestures |
| `gestures.py` | Six scale-free face/head features relative to a relaxed baseline; MediaPipe webcam adapter |
| `fatigue.py` | Mode advice with hysteresis; hourly report |
| `engine.py` | Match, speak, log, adapt; mutes the mic while speaking; arming; retry detection |
| `notify.py` | ntfy.sh push notifications for home phrases |
| `voice_clone.py` | Optional ElevenLabs clone, gated on recorded consent |
| `evaluate.py` | Leave-one-out accuracy on real recordings from a CSV manifest |
| `simulate.py` | Synthetic words and gestures for the demo and tests (not patient data) |

**Telling cues from conversation.** A teacher talks all day, so most sounds the
mic hears are not cue attempts. Encore only counts a rejected sound as a missed
cue if the user retries within 6 seconds (people repeat themselves when not
understood) or if they had armed it. Everything else is logged as background
and kept out of the monitor and report.

## Results so far (synthetic data, averaged over 8 random seeds)

`python -m encore demo` simulates a school day using the case's own numbers
(speech x1.40 slower by 4 PM) plus articulatory undershoot, a known feature of
dysarthric speech. Each hour has 8 cue attempts and 6 unrelated utterances.

| | Home mode | Classroom mode (arm cue) |
|---|---|---|
| Cue works at 9 AM (one retry allowed) | 100% | 100% |
| Cue works at 4 PM (one retry allowed) | 92% | 94% |
| Unrelated talk wrongly triggering a phrase | 13% (51 of 384) | 0% (0 of 384) |
| Mode advice | switches once, to hybrid, 1-3 PM | switches once, 1-3 PM |

On a single attempt with no retry, the matcher recognises 100% of cues at 9 AM
and 84% at 4 PM. Gesture cues: 100% at full movement, 12% once the movement
shrinks to 40% strength, 100% again after re-recording.

**The tradeoff we found.** Matching loose enough to work on a tired 4 PM voice
also lets roughly 1 in 8 unrelated utterances through. Tightening it cuts false
triggers to near zero but drops 4 PM recognition to 30-55%. No single setting
fixes both, so Encore offers two modes: home mode (sensitive; little background
talk, and a false trigger is low-stakes) and classroom mode, where voice cues
only fire for 6 seconds after an arm cue.

**Why classroom mode reaches 0%.** The arm cue is a gesture (a long, deliberate
blink in the demo), so speech alone can't arm it. That 0% holds by design in the
simulation. In real use the arm gesture must be one Alex doesn't make naturally
while talking. A brow raise, for example, would be a bad choice for an expressive
teacher. Test it during a real lesson before relying on it.

**About the mode advice.** The monitor looks at the last 16 cue attempts and needs
3 agreeing checks before switching (4, with a margin, to switch back). That gives
one clean switch per day in every simulated run, at the cost of reacting over
about two hours of use rather than minutes. That suits gradual daily fatigue but
would be slow to notice a sudden bad day.

These numbers show the pipeline behaves sensibly. They are not evidence of
accuracy on real dysarthric speech. That is what the next section is for.

## Testing on real data

`evaluate` runs leave-one-out matching within each speaker:

```bash
python -m encore evaluate manifest.csv
```

`manifest.csv` columns: `path,label,speaker,group`. For **VOC-ALS** (Synapse,
1,224 recordings, 102 people with ALS and 51 controls), use the task (vowel or
syllable) as `label`, the participant ID as `speaker`, and the ALSFRS-R speech
score or dysarthria severity as `group`. The output shows whether personal
templates keep working as severity increases, which is the core claim.

### Recording your team's voices

`collect` walks one person through recording each cue word three times in a
normal voice, then three times imitating a tired, slurred voice. It saves the
clips and builds the manifest automatically:

```bash
python -m encore collect --speaker aisha
python -m encore collect --speaker ben          # repeat for each teammate
python -m encore evaluate team_recordings/manifest.csv --compare normal slurred
```

`--compare` prints the three numbers that tell Encore's story:

| Scenario | What it shows |
|---|---|
| cues taught fresh, used fresh | does recognition work at all? |
| cues taught fresh, used tired | what happens as the voice fades through the day (Alex at 4 PM) |
| cues re-taught tired, used tired | does re-recording a cue rescue it? |

Be upfront with judges that imitated slurring is not real dysarthria. It is a
reasonable stand-in for a weekend, and VOC-ALS is the next step.

## Demo script for the expo table (about 3 minutes)

1. **The problem in Alex's numbers** (30 s): 165 to 118 words per minute between
   9 AM and 4 PM; students asking him to repeat; the recap he stopped doing.
2. **Live** (60 s): a teammate records "Here's today's recap" clearly, trains a
   mumbled cue, then mumbles it. Encore plays the clear recording with a caption.
   Then a head tilt triggers "I'm home and I'm okay" and a phone buzzes with the notification.
3. **It adapts, and we tested it honestly** (45 s): run `python -m encore demo`.
   Point at the afternoon decline, the single hybrid switch, the home vs classroom
   false-trigger comparison, and the gesture re-recording result.
4. **Why it won't end up in a drawer** (30 s): he chose the phrases; it's a
   teaching tool, not a patient device; it uses a laptop and webcam he already has.

Backup if the live mic fails in a loud room: `python -m encore demo`.

## Feasibility

Runs on a laptop or Raspberry Pi with a USB mic and webcam he already has. No
cloud required for core use; recordings stay on the device. Software-only, it is
a communication aid; Ontario's Assistive Devices Program funds some AAC devices,
and the path would be pairing with an AAC clinic and SLP rather than a new
regulated medical device. Matching one cue against 25 phrases with 3 examples each takes about 25 ms on a laptop.

## Limitations and ethics

- Not validated on real dysarthric speech yet (see Testing on real data).
- Loud rooms and bad lighting reduce accuracy. In home mode about 1 in 8 unrelated
  utterances triggered a phrase in simulation; classroom use should use a gesture arm cue.
- Notifications are a convenience, not a medical alert system: they need a cue
  in range of the mic or camera and do not detect falls.
- Voice cloning requires the voice owner's recorded consent and only uses his
  own Encore recordings. Recorded phrases always play the real recording.
- ntfy topics are public by name; use a long random topic.
- The report is a proxy signal from cue matching, not a clinical measure.

## Sponsor prizes this can opt into

- **Best Use of ElevenLabs**: `clone-consent`, `clone-create`, `clone-say`; the engine falls back to the clone for phrases with no recording.
- **Best 3D Print**: a printed clip that holds a phone or webcam at face level on a lectern or kitchen counter.
