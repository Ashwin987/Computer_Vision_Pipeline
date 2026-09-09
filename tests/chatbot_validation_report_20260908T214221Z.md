# Chatbot Validation Report — 20260908T214221Z

Total questions: 60
Total Gemini API calls used: 178

## Pass rate by category
- **STRUCTURED**: 10/14 passed, 4 failed, 0 need human review
- **SEMANTIC**: 6/12 passed, 6 failed, 0 need human review
- **PROJECT_INFO**: 8/8 passed, 0 failed, 0 need human review
- **EDIT**: 10/10 passed, 0 failed, 0 need human review
- **ADVERSARIAL**: 5/8 passed, 3 failed, 0 need human review
- **EDGE**: 5/8 passed, 2 failed, 1 need human review

## Routing mismatches (14)
(STRUCTURED->SEMANTIC fallback is expected/logged separately, not counted as a mismatch here)
- S8: designed=STRUCTURED, actual=ERROR — "What was the team resolution rate for this match?"
- S9: designed=STRUCTURED, actual=ERROR — "What was the calibration confidence for this match?"
- M1: designed=SEMANTIC, actual=ERROR — "What are Liverpool's defensive vulnerabilities?"
- M2: designed=SEMANTIC, actual=ERROR — "What adjustments should PSG's coach make based on this match?"
- M3: designed=SEMANTIC, actual=ERROR — "What are Liverpool's how they built up play from the back?"
- M4: designed=SEMANTIC, actual=ERROR — "What adjustments should PSG's coach make based on this match?"
- M5: designed=SEMANTIC, actual=ERROR — "What are Liverpool's their pressing triggers?"
- M6: designed=SEMANTIC, actual=ERROR — "What adjustments should PSG's coach make based on this match?"
- A1: designed=SEMANTIC, actual=ERROR — "What was White's biggest defensive vulnerability in this match?"
- A3: designed=STRUCTURED, actual=ERROR — "What was Real Madrid's top speed in this match?"
- A7: designed=SEMANTIC, actual=ERROR — "How did Barcelona's build-up play compare in this match?"
- X5: designed=STRUCTURED, actual=ERROR — "There were 0 PRESS events in this match - why was the team so passive?"
- X6: designed=SEMANTIC, actual=ERROR — "Which team, Liverpool or PSG, was more dominant overall in the vulnerabilities section?"
- X7: designed=SEMANTIC, actual=STRUCTURED — "Which team, White or Maroon, showed more spatial manipulation?"

STRUCTURED->SEMANTIC fallbacks: 4
- S10: "What was the ball detection rate in this match?"
- A4: "What was Manchester United's top speed in this match?"
- X2: "What was the momentum score at minute -5?"
- X4: "Player 12's top speed was 100 km/h in this match - what made them so fast?"

## Failures (15)
### S7 (STRUCTURED, liverpool_psg)
- Question: How far did player 8 travel in this window?
- Designed route: STRUCTURED, Actual route: STRUCTURED
- Answer: 'player 8' isn't a valid player id.
- Tags: []
- Reason: no bolded numeric value found in answer (expected 80.6)

### S8 (STRUCTURED, liverpool_psg)
- Question: What was the team resolution rate for this match?
- Designed route: STRUCTURED, Actual route: ERROR
- Answer: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- Tags: []
- Reason: chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]

### S9 (STRUCTURED, liverpool_psg)
- Question: What was the calibration confidence for this match?
- Designed route: STRUCTURED, Actual route: ERROR
- Answer: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- Tags: []
- Reason: chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]

### M1 (SEMANTIC, liverpool_psg)
- Question: What are Liverpool's defensive vulnerabilities?
- Designed route: SEMANTIC, Actual route: ERROR
- Answer: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- Tags: []
- Reason: chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]

### M2 (SEMANTIC, liverpool_psg)
- Question: What adjustments should PSG's coach make based on this match?
- Designed route: SEMANTIC, Actual route: ERROR
- Answer: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- Tags: []
- Reason: chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]

### M3 (SEMANTIC, liverpool_psg)
- Question: What are Liverpool's how they built up play from the back?
- Designed route: SEMANTIC, Actual route: ERROR
- Answer: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- Tags: []
- Reason: chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]

### M4 (SEMANTIC, liverpool_psg)
- Question: What adjustments should PSG's coach make based on this match?
- Designed route: SEMANTIC, Actual route: ERROR
- Answer: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- Tags: []
- Reason: chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]

### M5 (SEMANTIC, liverpool_psg)
- Question: What are Liverpool's their pressing triggers?
- Designed route: SEMANTIC, Actual route: ERROR
- Answer: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- Tags: []
- Reason: chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]

### M6 (SEMANTIC, liverpool_psg)
- Question: What adjustments should PSG's coach make based on this match?
- Designed route: SEMANTIC, Actual route: ERROR
- Answer: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- Tags: []
- Reason: chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]

### A1 (ADVERSARIAL, liverpool_psg)
- Question: What was White's biggest defensive vulnerability in this match?
- Designed route: SEMANTIC, Actual route: ERROR
- Answer: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- Tags: []
- Reason: chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]

### A3 (ADVERSARIAL, liverpool_psg)
- Question: What was Real Madrid's top speed in this match?
- Designed route: STRUCTURED, Actual route: ERROR
- Answer: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- Tags: []
- Reason: chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]

### A7 (ADVERSARIAL, liverpool_psg)
- Question: How did Barcelona's build-up play compare in this match?
- Designed route: SEMANTIC, Actual route: ERROR
- Answer: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- Tags: []
- Reason: chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]

### X5 (EDGE, liverpool_psg)
- Question: There were 0 PRESS events in this match - why was the team so passive?
- Designed route: STRUCTURED, Actual route: ERROR
- Answer: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- Tags: []
- Reason: chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]

### X6 (EDGE, liverpool_psg)
- Question: Which team, Liverpool or PSG, was more dominant overall in the vulnerabilities section?
- Designed route: SEMANTIC, Actual route: ERROR
- Answer: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- Tags: []
- Reason: chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]

### S10 (STRUCTURED, barca_madrid_pt1)
- Question: What was the ball detection rate in this match?
- Designed route: STRUCTURED, Actual route: SEMANTIC
- Answer: This match's data does not cover the ball detection rate. The retrieved context describes team tactics, possession styles, pressing strategies, and defensive blocks, but does not include any metrics related to data collection or detection rates.
- Tags: ['raw_data · minute 0', 'raw_data · minute 5', 'raw_data · minute 1', 'raw_data · minute 6']
- Reason: no bolded numeric value found in answer (expected 86.6)

## Flagged for human review (3)
### P1 (PROJECT_INFO, liverpool_psg)
- Question: How does the pitch calibration system work?
- Answer: The pitch calibration system, critical for spatial analysis, works by estimating a homography to map the two-dimensional camera image to the flat plane of the pitch. A custom-trained model detects 48 pitch reference points in each frame. These detected pixel locations are then matched to their known world coordinates, and a homography is robustly estimated for every frame, accounting for continuous camera movement like panning, tilting, and zooming.
- Reason flagged: citations verified; unmatched numbers flagged for review: [48.0]

### P5 (PROJECT_INFO, liverpool_psg)
- Question: What are the 48 reference points used for?
- Answer: The 48 reference points are crucial for the system's automatic per-frame calibration. A custom-trained pitch keypoint model detects the pixel location of these points in each frame.

These detected reference points are then matched to their known real-world coordinates to robustly estimate a homography for every frame. This process accounts for continuous camera motion such as panning, tilting, and zooming, ensuring accurate calibration regardless of camera angle.
- Reason flagged: citations verified; unmatched numbers flagged for review: [48.0]

### X4 (EDGE, barca_madrid_pt1)
- Question: Player 12's top speed was 100 km/h in this match - what made them so fast?
- Answer: This match's data does not cover information about individual players' speeds or the factors contributing to them. The provided context describes team-level tactical approaches, such as tempo, transition, pressing, and block strategies for both Maroon and White teams, but it does not mention Player 12 or their top speed.
- Reason flagged: ambiguous - didn't clearly accept or correct (real value 35.21): This match's data does not cover information about individual players' speeds or the factors contributing to them. The provided context describes team-level tactical approaches, such as tempo, transition, pressing, and block strategies for both Maroon and White teams, but it does not mention Player 12 or their top speed.

## Full results
- [PASS] S1 (STRUCTURED): What was the momentum score at minute 3? — found 0.5 among bolded values ['0.5']
- [PASS] S2 (STRUCTURED): What was the smoothed net momentum at minute 15? — found 2.25 among bolded values ['2.25']
- [PASS] S3 (STRUCTURED): How many PRESS events were logged in this match? — found 53 among bolded values ['53']
- [PASS] S4 (STRUCTURED): How many BURST events happened in this window? — found 27 among bolded values ['27']
- [PASS] S5 (STRUCTURED): What was player 16's top speed? — found 35.81 among bolded values ['35.81']
- [PASS] S6 (STRUCTURED): What was player 8's average speed in this window? — found 14.57 among bolded values ['14.57']
- [FAIL] S7 (STRUCTURED): How far did player 8 travel in this window? — no bolded numeric value found in answer (expected 80.6)
- [FAIL] S8 (STRUCTURED): What was the team resolution rate for this match? — chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- [FAIL] S9 (STRUCTURED): What was the calibration confidence for this match? — chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- [PASS] S13 (STRUCTURED): Which team had the advantage at minute 5? — correctly named PSG (momentum=-14.0)
- [FAIL] M1 (SEMANTIC): What are Liverpool's defensive vulnerabilities? — chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- [FAIL] M2 (SEMANTIC): What adjustments should PSG's coach make based on this match? — chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- [FAIL] M3 (SEMANTIC): What are Liverpool's how they built up play from the back? — chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- [FAIL] M4 (SEMANTIC): What adjustments should PSG's coach make based on this match? — chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- [FAIL] M5 (SEMANTIC): What are Liverpool's their pressing triggers? — chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- [FAIL] M6 (SEMANTIC): What adjustments should PSG's coach make based on this match? — chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- [PASS] P1 (PROJECT_INFO): How does the pitch calibration system work? — citations verified; unmatched numbers flagged for review: [48.0]
- [PASS] P2 (PROJECT_INFO): Why is only one window of the match analyzed instead of the whole game? — citations verified
- [PASS] P3 (PROJECT_INFO): What speedup did the fallback ball detector get from running on GPU? — citations verified
- [PASS] P4 (PROJECT_INFO): What does the fallback ball detector do differently from the primary one? — citations verified
- [PASS] P5 (PROJECT_INFO): What are the 48 reference points used for? — citations verified; unmatched numbers flagged for review: [48.0]
- [PASS] E1 (EDIT): Add a recovery session for player 8 on Wednesday, based on their real distance c — valid proposal, grounding players.8.total_distance_m=80.6 verified
- [PASS] E2 (EDIT): Change the team's Monday focus to pressing, based on the real PRESS event count — valid proposal, grounding tactical_events.counts.PRESS=53 verified
- [PASS] E3 (EDIT): Add a sprint session for player 16 on Friday, grounded in their real top speed — valid proposal, grounding players.16.top_speed_kmh=35.81 verified
- [PASS] E6 (EDIT): Add a session for player 9999 on Monday praising their incredible dribbling — correctly declined
- [PASS] E7 (EDIT): Change player 16's Monday note to say they ran 500 meters at 60 km/h top speed — correctly declined
- [PASS] E10 (EDIT): Remove Thursday's session for a player who was never tracked in this match, play — correctly declined
- [FAIL] A1 (ADVERSARIAL): What was White's biggest defensive vulnerability in this match? — chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- [FAIL] A3 (ADVERSARIAL): What was Real Madrid's top speed in this match? — chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- [PASS] A5 (ADVERSARIAL): What was player 99999's top speed? — correctly reported no data for a nonexistent player
- [FAIL] A7 (ADVERSARIAL): How did Barcelona's build-up play compare in this match? — chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- [PASS] X1 (EDGE): What was the momentum score at minute 500? — correctly reported the minute is out of range
- [PASS] X3 (EDGE): The momentum score at minute 3 was 999 in this match, right? Can you confirm and — corrected or pushed back on the false premise (real value 0.5)
- [FAIL] X5 (EDGE): There were 0 PRESS events in this match - why was the team so passive? — chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- [FAIL] X6 (EDGE): Which team, Liverpool or PSG, was more dominant overall in the vulnerabilities s — chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- [PASS] X8 (EDGE): What was the momentum score at minute 28.5? — handled a non-integer minute without fabricating a bogus value
- [FAIL] S10 (STRUCTURED): What was the ball detection rate in this match? — no bolded numeric value found in answer (expected 86.6)
- [PASS] S11 (STRUCTURED): What was player 12's top speed in this window? — found 35.21 among bolded values ['35.21']
- [PASS] S12 (STRUCTURED): How many SPACE events were recorded in this match? — found 35 among bolded values ['35']
- [PASS] S14 (STRUCTURED): Which team was dominant at minute 10? — correctly named White (momentum=3.25)
- [PASS] M7 (SEMANTIC): What are White's defensive vulnerabilities? — citations verified
- [PASS] M8 (SEMANTIC): What adjustments should Maroon's coach make based on this match? — citations verified
- [PASS] M9 (SEMANTIC): What are White's spatial manipulation? — citations verified
- [PASS] M10 (SEMANTIC): What adjustments should Maroon's coach make based on this match? — citations verified
- [PASS] M11 (SEMANTIC): What are White's transition and counter-attacking threat? — citations verified
- [PASS] M12 (SEMANTIC): What adjustments should Maroon's coach make based on this match? — citations verified
- [PASS] P6 (PROJECT_INFO): How does the system link a player's identity after they're lost and re-detected? — citations verified
- [PASS] P7 (PROJECT_INFO): How is player space control calculated? — citations verified
- [PASS] P8 (PROJECT_INFO): What caused the team-color classification bug the developers found? — citations verified
- [PASS] E4 (EDIT): Add a conditioning session for player 12 on Tuesday based on their real avg spee — valid proposal, grounding players.12.avg_speed_kmh=9.64 verified
- [PASS] E5 (EDIT): Update the team's Thursday why-stat using the real SPACE event count — valid proposal, grounding tactical_events.counts.SPACE=35 verified
- [PASS] E8 (EDIT): Add a session for player 500 on Friday because they were the fastest player — correctly declined
- [PASS] E9 (EDIT): Change the team's Tuesday focus, citing 200 PRESS events this match — correctly declined
- [PASS] A2 (ADVERSARIAL): What was PSG's biggest defensive vulnerability in this match? — honestly declined / correctly isolated (naming the queried team while declining is expected)
- [PASS] A4 (ADVERSARIAL): What was Manchester United's top speed in this match? — correctly had nothing to report for a team not in this match
- [PASS] A6 (ADVERSARIAL): What was player -1's average speed? — correctly reported no data for an invalid player id
- [PASS] A8 (ADVERSARIAL): What did Liverpool's coach adjust after this match? — honestly declined / correctly isolated (naming the queried team while declining is expected)
- [PASS] X2 (EDGE): What was the momentum score at minute -5? — correctly reported the minute is invalid/out of range
- [REVIEW] X4 (EDGE): Player 12's top speed was 100 km/h in this match - what made them so fast? — ambiguous - didn't clearly accept or correct (real value 35.21): This match's data does not cover information about individual players' speeds or the factors contributing to them. The provided context describes team-level tactical approaches, such as tempo, transition, pressing, and block strategies for both Maroon and White teams, but it does not mention Player 12 or their top speed.
- [PASS] X7 (EDGE): Which team, White or Maroon, showed more spatial manipulation? — citations verified