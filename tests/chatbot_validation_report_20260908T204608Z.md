# Chatbot Validation Report — 20260908T204608Z

Total questions: 60
Total Gemini API calls used: 95

## Pass rate by category
- **STRUCTURED**: 10/14 passed, 4 failed, 0 need human review
- **SEMANTIC**: 10/12 passed, 2 failed, 0 need human review
- **PROJECT_INFO**: 8/8 passed, 0 failed, 0 need human review
- **EDIT**: 10/10 passed, 0 failed, 0 need human review
- **ADVERSARIAL**: 4/8 passed, 2 failed, 2 need human review
- **EDGE**: 6/8 passed, 0 failed, 2 need human review

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

## Failures (8)
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
- Reason: no bolded numeric value found in answer (expected 82.2)

### S9 (STRUCTURED, liverpool_psg)
- Question: What was the calibration confidence for this match?
- Designed route: STRUCTURED, Actual route: ERROR
- Answer: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- Tags: []
- Reason: no bolded numeric value found in answer (expected 0.9009)

### S10 (STRUCTURED, barca_madrid_pt1)
- Question: What was the ball detection rate in this match?
- Designed route: STRUCTURED, Actual route: SEMANTIC
- Answer: This match's data does not cover the ball detection rate. The provided context describes tactical aspects such as possession, pressing intensity, defensive blocks, and build-up styles for both teams (White and Maroon). It does not include specific statistical metrics like ball detection rate.
- Tags: ['raw_data · minute 0', 'raw_data · minute 5', 'raw_data · minute 1', 'raw_data · minute 6']
- Reason: no bolded numeric value found in answer (expected 86.6)

### M7 (SEMANTIC, barca_madrid_pt1)
- Question: What are White's defensive vulnerabilities?
- Designed route: SEMANTIC, Actual route: SEMANTIC
- Answer: White's aggressive high defensive line is susceptible to well-timed long balls or quick passes over the top, especially against fast, channel-running strikers. Their high press can be bypassed by quick, incisive passing combinations, leaving the midfield exposed and creating numerical disadvantages. Furthermore, the overlapping runs of their fullbacks can leave their defensive flanks vulnerable to rapid counter-attacks if possession is lost high up the pitch.
- Tags: ['coach report · White Tactical Profile · Vulnerabilities', 'coach report · Maroon Tactical Profile · Vulnerabilities', 'raw_data · minute 0', 'raw_data · minute 4']
- Reason: citation(s) don't match real data: ['coach report · White Tactical Profile · Vulnerabilities', 'coach report · Maroon Tactical Profile · Vulnerabilities']

### M9 (SEMANTIC, barca_madrid_pt1)
- Question: What are White's spatial manipulation?
- Designed route: SEMANTIC, Actual route: SEMANTIC
- Answer: White effectively manipulates space by deploying two players to occupy the half-spaces, creating crucial passing lanes and dilemmas for the opposition. This strategy facilitates central penetration and links play between midfield and attack. Complementing this, their fullbacks consistently overlap to provide natural width, stretching the opponent's defensive block and creating crossing opportunities. This combined approach allows White to attack both centrally and from wide areas, preventing the opposition from becoming too compact.
- Tags: ['coach report · White Tactical Profile · Spatial Manipulation', 'raw_data · minute 0', 'raw_data · minute 5', 'raw_data · minute 7']
- Reason: citation(s) don't match real data: ['coach report · White Tactical Profile · Spatial Manipulation']

### A2 (ADVERSARIAL, barca_madrid_pt1)
- Question: What was PSG's biggest defensive vulnerability in this match?
- Designed route: SEMANTIC, Actual route: SEMANTIC
- Answer: This match's data doesn't cover PSG's defensive vulnerabilities. The provided context analyzes the tactical profiles and vulnerabilities of "White" and "Maroon" teams, not PSG.
- Tags: ['coach report · White Tactical Profile · Vulnerabilities', 'coach report · Maroon Tactical Profile · Vulnerabilities', 'coach report · Maroon Tactical Profile · Defensive Mechanics', 'coach report · DATA-DRIVEN COACHING ADJUSTMENTS']
- Reason: leaked other-match fact(s) into this match's answer: ['psg']

### A8 (ADVERSARIAL, barca_madrid_pt1)
- Question: What did Liverpool's coach adjust after this match?
- Designed route: SEMANTIC, Actual route: SEMANTIC
- Answer: This match's data does not cover adjustments made by Liverpool's coach. The retrieved context only discusses data-driven coaching adjustments for teams named Maroon and White. There is no mention of Liverpool or their coaching staff in the provided information.
- Tags: ['coach report · DATA-DRIVEN COACHING ADJUSTMENTS', 'raw_data · minute 5', 'raw_data · minute 2', 'raw_data · minute 6']
- Reason: leaked other-match fact(s) into this match's answer: ['liverpool']

## Flagged for human review (6)
### P1 (PROJECT_INFO, liverpool_psg)
- Question: How does the pitch calibration system work?
- Answer: The pitch calibration system works by detecting 48 pitch reference points in each frame using a custom-trained keypoint model. These detected pixel locations are then matched to their known world coordinates. For each frame, a homography is estimated from these correspondences, establishing the mapping between the 2D camera image and the flat pitch plane. This homography is recomputed for every frame to account for continuous camera movement.
- Reason flagged: citations verified; unmatched numbers flagged for review: [48.0]

### P5 (PROJECT_INFO, liverpool_psg)
- Question: What are the 48 reference points used for?
- Answer: The 48 reference points are detected by a custom-trained pitch keypoint model for each frame. These points enable automatic per-frame calibration of the broadcast footage. For every frame, the detected reference points are matched to their known real-world coordinates on the pitch, allowing for the robust estimation of a homography. This homography accounts for continuous camera motion, including panning, tilting, and zooming.
- Reason flagged: citations verified; unmatched numbers flagged for review: [48.0]

### A1 (ADVERSARIAL, liverpool_psg)
- Question: What was White's biggest defensive vulnerability in this match?
- Answer: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- Reason flagged: no leaked facts detected, but no clear honest-decline phrase either - human review

### A7 (ADVERSARIAL, liverpool_psg)
- Question: How did Barcelona's build-up play compare in this match?
- Answer: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- Reason flagged: no leaked facts detected, but no clear honest-decline phrase either - human review

### X4 (EDGE, barca_madrid_pt1)
- Question: Player 12's top speed was 100 km/h in this match - what made them so fast?
- Answer: This match's data does not cover player-specific statistics like Player 12's top speed, nor does it provide reasons for any individual player's speed. The context describes team tactics and tempo, such as Maroon's "fast direct tempo" and "fast vertical transition," but not individual player attributes.
- Reason flagged: ambiguous - didn't clearly accept or correct (real value 35.21): This match's data does not cover player-specific statistics like Player 12's top speed, nor does it provide reasons for any individual player's speed. The context describes team tactics and tempo, such as Maroon's "fast direct tempo" and "fast vertical transition," but not individual player attributes.

### X5 (EDGE, liverpool_psg)
- Question: There were 0 PRESS events in this match - why was the team so passive?
- Answer: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- Reason flagged: ambiguous - didn't clearly accept or correct (real value 53): [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]

## Full results
- [PASS] S1 (STRUCTURED): What was the momentum score at minute 3? — found 0.5 among bolded values ['0.5']
- [PASS] S2 (STRUCTURED): What was the smoothed net momentum at minute 15? — found 2.25 among bolded values ['2.25']
- [PASS] S3 (STRUCTURED): How many PRESS events were logged in this match? — found 53 among bolded values ['53']
- [PASS] S4 (STRUCTURED): How many BURST events happened in this window? — found 27 among bolded values ['27']
- [PASS] S5 (STRUCTURED): What was player 16's top speed? — found 35.81 among bolded values ['35.81']
- [PASS] S6 (STRUCTURED): What was player 8's average speed in this window? — found 14.57 among bolded values ['14.57']
- [FAIL] S7 (STRUCTURED): How far did player 8 travel in this window? — no bolded numeric value found in answer (expected 80.6)
- [FAIL] S8 (STRUCTURED): What was the team resolution rate for this match? — no bolded numeric value found in answer (expected 82.2)
- [FAIL] S9 (STRUCTURED): What was the calibration confidence for this match? — no bolded numeric value found in answer (expected 0.9009)
- [FAIL] S10 (STRUCTURED): What was the ball detection rate in this match? — no bolded numeric value found in answer (expected 86.6)
- [PASS] S11 (STRUCTURED): What was player 12's top speed in this window? — found 35.21 among bolded values ['35.21']
- [PASS] S12 (STRUCTURED): How many SPACE events were recorded in this match? — found 35 among bolded values ['35']
- [PASS] S13 (STRUCTURED): Which team had the advantage at minute 5? — correctly named PSG (momentum=-14.0)
- [PASS] S14 (STRUCTURED): Which team was dominant at minute 10? — correctly named White (momentum=3.25)
- [PASS] M1 (SEMANTIC): What are Liverpool's defensive vulnerabilities? — citations verified
- [PASS] M2 (SEMANTIC): What adjustments should PSG's coach make based on this match? — citations verified
- [PASS] M3 (SEMANTIC): What are Liverpool's how they built up play from the back? — citations verified
- [PASS] M4 (SEMANTIC): What adjustments should PSG's coach make based on this match? — citations verified
- [PASS] M5 (SEMANTIC): What are Liverpool's their pressing triggers? — citations verified
- [PASS] M6 (SEMANTIC): What adjustments should PSG's coach make based on this match? — citations verified
- [FAIL] M7 (SEMANTIC): What are White's defensive vulnerabilities? — citation(s) don't match real data: ['coach report · White Tactical Profile · Vulnerabilities', 'coach report · Maroon Tactical Profile · Vulnerabilities']
- [PASS] M8 (SEMANTIC): What adjustments should Maroon's coach make based on this match? — citations verified
- [FAIL] M9 (SEMANTIC): What are White's spatial manipulation? — citation(s) don't match real data: ['coach report · White Tactical Profile · Spatial Manipulation']
- [PASS] M10 (SEMANTIC): What adjustments should Maroon's coach make based on this match? — citations verified
- [PASS] M11 (SEMANTIC): What are White's transition and counter-attacking threat? — citations verified
- [PASS] M12 (SEMANTIC): What adjustments should Maroon's coach make based on this match? — citations verified
- [PASS] P1 (PROJECT_INFO): How does the pitch calibration system work? — citations verified; unmatched numbers flagged for review: [48.0]
- [PASS] P2 (PROJECT_INFO): Why is only one window of the match analyzed instead of the whole game? — citations verified
- [PASS] P3 (PROJECT_INFO): What speedup did the fallback ball detector get from running on GPU? — citations verified
- [PASS] P4 (PROJECT_INFO): What does the fallback ball detector do differently from the primary one? — citations verified
- [PASS] P5 (PROJECT_INFO): What are the 48 reference points used for? — citations verified; unmatched numbers flagged for review: [48.0]
- [PASS] P6 (PROJECT_INFO): How does the system link a player's identity after they're lost and re-detected? — citations verified
- [PASS] P7 (PROJECT_INFO): How is player space control calculated? — citations verified
- [PASS] P8 (PROJECT_INFO): What caused the team-color classification bug the developers found? — citations verified
- [PASS] E1 (EDIT): Add a recovery session for player 8 on Wednesday, based on their real distance c — valid proposal, grounding players.8.total_distance_m=80.6 verified
- [PASS] E2 (EDIT): Change the team's Monday focus to pressing, based on the real PRESS event count — valid proposal, grounding tactical_events.counts.PRESS=53 verified
- [PASS] E3 (EDIT): Add a sprint session for player 16 on Friday, grounded in their real top speed — valid proposal, grounding players.16.top_speed_kmh=35.81 verified
- [PASS] E4 (EDIT): Add a conditioning session for player 12 on Tuesday based on their real avg spee — valid proposal, grounding players.12.avg_speed_kmh=9.64 verified
- [PASS] E5 (EDIT): Update the team's Thursday why-stat using the real SPACE event count — valid proposal, grounding tactical_events.counts.SPACE=35 verified
- [PASS] E6 (EDIT): Add a session for player 9999 on Monday praising their incredible dribbling — correctly declined
- [PASS] E7 (EDIT): Change player 16's Monday note to say they ran 500 meters at 60 km/h top speed — correctly declined
- [PASS] E8 (EDIT): Add a session for player 500 on Friday because they were the fastest player — correctly declined
- [PASS] E9 (EDIT): Change the team's Tuesday focus, citing 200 PRESS events this match — correctly declined
- [PASS] E10 (EDIT): Remove Thursday's session for a player who was never tracked in this match, play — correctly declined
- [REVIEW] A1 (ADVERSARIAL): What was White's biggest defensive vulnerability in this match? — no leaked facts detected, but no clear honest-decline phrase either - human review
- [FAIL] A2 (ADVERSARIAL): What was PSG's biggest defensive vulnerability in this match? — leaked other-match fact(s) into this match's answer: ['psg']
- [PASS] A3 (ADVERSARIAL): What was Real Madrid's top speed in this match? — correctly had nothing to report for a team not in this match
- [PASS] A4 (ADVERSARIAL): What was Manchester United's top speed in this match? — correctly had nothing to report for a team not in this match
- [PASS] A5 (ADVERSARIAL): What was player 99999's top speed? — correctly reported no data for a nonexistent player
- [PASS] A6 (ADVERSARIAL): What was player -1's average speed? — correctly reported no data for an invalid player id
- [REVIEW] A7 (ADVERSARIAL): How did Barcelona's build-up play compare in this match? — no leaked facts detected, but no clear honest-decline phrase either - human review
- [FAIL] A8 (ADVERSARIAL): What did Liverpool's coach adjust after this match? — leaked other-match fact(s) into this match's answer: ['liverpool']
- [PASS] X1 (EDGE): What was the momentum score at minute 500? — correctly reported the minute is out of range
- [PASS] X2 (EDGE): What was the momentum score at minute -5? — correctly reported the minute is invalid/out of range
- [PASS] X3 (EDGE): The momentum score at minute 3 was 999 in this match, right? Can you confirm and — corrected or pushed back on the false premise (real value 0.5)
- [REVIEW] X4 (EDGE): Player 12's top speed was 100 km/h in this match - what made them so fast? — ambiguous - didn't clearly accept or correct (real value 35.21): This match's data does not cover player-specific statistics like Player 12's top speed, nor does it provide reasons for any individual player's speed. The context describes team tactics and tempo, such as Maroon's "fast direct tempo" and "fast vertical transition," but not individual player attributes.
- [REVIEW] X5 (EDGE): There were 0 PRESS events in this match - why was the team so passive? — ambiguous - didn't clearly accept or correct (real value 53): [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- [PASS] X6 (EDGE): Which team, Liverpool or PSG, was more dominant overall in the vulnerabilities s — citations verified
- [PASS] X7 (EDGE): Which team, White or Maroon, showed more spatial manipulation? — citations verified
- [PASS] X8 (EDGE): What was the momentum score at minute 28.5? — handled a non-integer minute without fabricating a bogus value