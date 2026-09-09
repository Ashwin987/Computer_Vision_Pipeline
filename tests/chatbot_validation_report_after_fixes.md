# Chatbot Validation Report — 20260908T222442Z

Total questions: 60
Total Gemini API calls used: 158

## Pass rate by category
- **STRUCTURED**: 14/14 passed, 0 failed, 0 need human review
- **SEMANTIC**: 12/12 passed, 0 failed, 0 need human review
- **PROJECT_INFO**: 8/8 passed, 0 failed, 0 need human review
- **EDIT**: 10/10 passed, 0 failed, 0 need human review
- **ADVERSARIAL**: 8/8 passed, 0 failed, 0 need human review
- **EDGE**: 6/8 passed, 1 failed, 1 need human review

## Routing mismatches (2)
(STRUCTURED->SEMANTIC fallback is expected/logged separately, not counted as a mismatch here)
- M5: designed=SEMANTIC, actual=STRUCTURED — "What are Liverpool's their pressing triggers?"
- X7: designed=SEMANTIC, actual=STRUCTURED — "Which team, White or Maroon, showed more spatial manipulation?"

STRUCTURED->SEMANTIC fallbacks: 5
- A3: "What was Real Madrid's top speed in this match?"
- X5: "There were 0 PRESS events in this match - why was the team so passive?"
- A4: "What was Manchester United's top speed in this match?"
- X2: "What was the momentum score at minute -5?"
- X4: "Player 12's top speed was 100 km/h in this match - what made them so fast?"

## Failures (1)
### X5 (EDGE, liverpool_psg)
- Question: There were 0 PRESS events in this match - why was the team so passive?
- Designed route: STRUCTURED, Actual route: SEMANTIC
- Answer: The provided context does not support the assertion of 0 PRESS events in this match. In fact, Liverpool's defensive mechanics profile states their "pressing intensity was notably high at 8/10," indicating a fierce commitment to winning the ball back and employing "gegenpressing."

While one specific segment mentions Liverpool defending with "low pressing intensity" when PSG held possession in the attacking third, this does not suggest an absence of all press events or overall passivity. Instead, it highlights a generally high pressing intensity, particularly upon losing possession.
- Tags: ['coach report · Liverpool Tactical Profile · Defensive Mechanics', 'coach report · Liverpool Tactical Profile · Possession & Territory', 'raw_data · minute 3', 'coach report · Tactical Diagnosis & Match Flow']
- Reason: accepted false premise (0) without correcting to real value (53)

## Flagged for human review (3)
### P1 (PROJECT_INFO, liverpool_psg)
- Question: How does the pitch calibration system work?
- Answer: The pitch calibration system operates by detecting 48 pitch reference points in each frame using a custom-trained keypoint model. The model predicts the pixel location of these visible reference points, which are then matched to their known world coordinates. For every frame, a homography is robustly estimated based on these correspondences, mapping the 2D camera image to the flat pitch plane. This ensures continuous calibration that accounts for camera panning, tilting, and zooming.
- Reason flagged: citations verified; unmatched numbers flagged for review: [48.0]

### P5 (PROJECT_INFO, liverpool_psg)
- Question: What are the 48 reference points used for?
- Answer: The 48 reference points are pitch keypoints detected per frame by a custom-trained model. Their primary purpose is to enable automatic per-frame calibration of the broadcast camera. This is achieved by matching the detected pixel locations of these visible reference points to their known real-world coordinates on the pitch, allowing for the robust estimation of a homography for each frame.
- Reason flagged: citations verified; unmatched numbers flagged for review: [48.0]

### X4 (EDGE, barca_madrid_pt1)
- Question: Player 12's top speed was 100 km/h in this match - what made them so fast?
- Answer: This match's data does not cover Player 12's top speed or what made them fast. The provided context only discusses team possession, tempo, and defensive strategies, without mentioning individual player metrics like top speed.
- Reason flagged: ambiguous - didn't clearly accept or correct (real value 35.21): This match's data does not cover Player 12's top speed or what made them fast. The provided context only discusses team possession, tempo, and defensive strategies, without mentioning individual player metrics like top speed.

## Full results
- [PASS] S1 (STRUCTURED): What was the momentum score at minute 3? — found 0.5 among bolded values ['0.5']
- [PASS] S2 (STRUCTURED): What was the smoothed net momentum at minute 15? — found 2.25 among bolded values ['2.25']
- [PASS] S3 (STRUCTURED): How many PRESS events were logged in this match? — found 53 among bolded values ['53']
- [PASS] S4 (STRUCTURED): How many BURST events happened in this window? — found 27 among bolded values ['27']
- [PASS] S5 (STRUCTURED): What was player 16's top speed? — found 35.81 among bolded values ['35.81']
- [PASS] S6 (STRUCTURED): What was player 8's average speed in this window? — found 14.57 among bolded values ['14.57']
- [PASS] S7 (STRUCTURED): How far did player 8 travel in this window? — found 80.6 among bolded values ['80.6']
- [PASS] S8 (STRUCTURED): What was the team resolution rate for this match? — found 82.2 among bolded values ['82.2']
- [PASS] S9 (STRUCTURED): What was the calibration confidence for this match? — found 0.9009 among bolded values ['0.9009']
- [PASS] S13 (STRUCTURED): Which team had the advantage at minute 5? — correctly named PSG (momentum=-14.0)
- [PASS] M1 (SEMANTIC): What are Liverpool's defensive vulnerabilities? — citations verified
- [PASS] M2 (SEMANTIC): What adjustments should PSG's coach make based on this match? — citations verified
- [PASS] M3 (SEMANTIC): What are Liverpool's how they built up play from the back? — citations verified
- [PASS] M4 (SEMANTIC): What adjustments should PSG's coach make based on this match? — citations verified
- [PASS] M5 (SEMANTIC): What are Liverpool's their pressing triggers? — citations verified
- [PASS] M6 (SEMANTIC): What adjustments should PSG's coach make based on this match? — citations verified
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
- [PASS] A1 (ADVERSARIAL): What was White's biggest defensive vulnerability in this match? — honestly declined / correctly isolated (naming the queried team while declining is expected)
- [PASS] A3 (ADVERSARIAL): What was Real Madrid's top speed in this match? — correctly had nothing to report for a team not in this match
- [PASS] A5 (ADVERSARIAL): What was player 99999's top speed? — correctly reported no data for a nonexistent player
- [PASS] A7 (ADVERSARIAL): How did Barcelona's build-up play compare in this match? — honestly declined / correctly isolated (naming the queried team while declining is expected)
- [PASS] X1 (EDGE): What was the momentum score at minute 500? — correctly reported the minute is out of range
- [PASS] X3 (EDGE): The momentum score at minute 3 was 999 in this match, right? Can you confirm and — corrected or pushed back on the false premise (real value 0.5)
- [FAIL] X5 (EDGE): There were 0 PRESS events in this match - why was the team so passive? — accepted false premise (0) without correcting to real value (53)
- [PASS] X6 (EDGE): Which team, Liverpool or PSG, was more dominant overall in the vulnerabilities s — citations verified
- [PASS] X8 (EDGE): What was the momentum score at minute 28.5? — handled a non-integer minute without fabricating a bogus value
- [PASS] S10 (STRUCTURED): What was the ball detection rate in this match? — found 86.6 among bolded values ['86.6']
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
- [REVIEW] X4 (EDGE): Player 12's top speed was 100 km/h in this match - what made them so fast? — ambiguous - didn't clearly accept or correct (real value 35.21): This match's data does not cover Player 12's top speed or what made them fast. The provided context only discusses team possession, tempo, and defensive strategies, without mentioning individual player metrics like top speed.
- [PASS] X7 (EDGE): Which team, White or Maroon, showed more spatial manipulation? — citations verified