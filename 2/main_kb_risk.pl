/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Modified version of main.pl that relies more on state of KB instead
    of just a list of actions.

    A list of actions is useful for planning but KB is much faster at
    analyzing percepts.
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

% These must be dynamic so that many different worlds might be created
% at runtime.
:- abolish(w_Wall/1).
:- abolish(w_Wumpus/1).
:- abolish(w_Gold/1).
:- abolish(w_Pit/1).
:- abolish(a_hunterAlive/0).
:- abolish(a_wumpusAlive/0).
:- abolish(a_perceiveScream/0).
:- abolish(a_perceiveBreeze/1).
:- abolish(a_perceiveStench/1).
:- abolish(a_perceiveGlitter/1).
:- abolish(a_hunterPosition/1).
:- abolish(a_hunterDirection/1).
:- abolish(a_hasGold/0).
:- abolish(a_hasArrow/0).
:- abolish(a_visited/1).
:- abolish(a_wumpusPosition/1).


:- dynamic([
  w_Wall/1,
  w_Wumpus/1,
  w_Gold/1,
  w_Pit/1,
  a_hunterAlive/0,
  a_wumpusAlive/0,
  a_perceiveScream/0,
  a_perceiveBreeze/1,
  a_perceiveStench/1,
  a_perceiveGlitter/1,
  a_hunterPosition/1,
  a_hunterDirection/1,
  a_hasGold/0,
  a_hasArrow/0,
  a_visited/1,
  a_wumpusPosition/1
]).

%CONSTANTS
maxNumberOfMoves(40).
maxNumberOfActionsPlanned(20).


% utils


directionChange(n,w).
directionChange(e,n).
directionChange(s,e).
directionChange(w,s).

% Hunter Position and Facing Direction on each situation
hunter(r(1,1),e,s0). %Start at cave entry r(1,1), facing east
%if action changes hunter position
hunter(R,D,do(A,S)) :- hunter(R0,D0,S), %get hunter info at last situation
    ( 
        (A = left, R = R0, directionChange(D0,D)); %turn left
        (A = right, R = R0, directionChange(D,D0)); %turn right
        (A = forward, D = D0, getForwardRoom(R0,D0,RN), !, (w_Wall(RN) -> R = R0 ; R = RN)) %go forward
    ).
%no actions that change hunter position happened
hunter(R,D,do(A,S)) :-
    hunter(R,D,S), %Position of hunter now same as before
    \+A = left,
    \+A = right,
    \+A = forward.


% dynamically updated state

a_hasGold :- false. % just to be explicit
a_hasArrow.
a_wumpusAlive. 
a_hunterAlive.
a_hunterPosition(r(1,1)).
a_hunterDirection(e).

resetWorld :-
    retractall(a_hasGold),
    asserta(a_hasArrow),
    asserta(a_wumpusAlive),
    asserta(a_hunterAlive),
    retractall(a_hunterPosition(_)),
    asserta(a_hunterPosition(r(1,1))),
    retractall(a_hunterDirection(_)),
    asserta(a_hunterDirection(e)),
    retractall(a_visited(_)),
    asserta(a_visited(r(1,1))),
    retractall(a_perceiveBreeze(_)),
    retractall(a_perceiveGlitter(_)),
    retractall(a_perceiveScream),
    retractall(a_perceiveStench(_)),
    retractall(a_wumpusPosition(_)).

update_hunterAlive(forward) :-
    a_hunterPosition(R),
    (w_Pit(R); (a_wumpusAlive, w_Wumpus(R))),
%    format('hunter died at ~w~n', R),
    retractall(a_hunterAlive).
update_hunterAlive(_).

update_hunterDirection(left) :- !,
    a_hunterDirection(PD),
    directionChange(PD, ND),
    retractall(a_hunterDirection(_)),
    asserta(a_hunterDirection(ND)).
update_hunterDirection(right) :- !,
    a_hunterDirection(PD),
    directionChange(ND, PD),
    retractall(a_hunterDirection(_)),
    asserta(a_hunterDirection(ND)).
update_hunterDirection(_).

update_hunterPosition(forward) :- !,
    a_hunterDirection(D),
    a_hunterPosition(R0),
    getForwardRoom(R0,D,RN), !, (w_Wall(RN) -> R = R0 ; R = RN),
    retractall(a_hunterPosition(_)),
    asserta(a_hunterPosition(R)),
    asserta(a_visited(R)).

update_hunterPosition(_).

update_wumpusAlive(shoot) :- !,
    a_hasArrow,
    a_hunterPosition(R0),
    a_hunterDirection(D0),
    w_Wumpus(RW),
    isFacing(R0,D0,RW),%If he is shot, he dies
    asserta(a_perceiveScream),
    retractall(a_wumpusAlive).
update_wumpusAlive(_).

update_hasArrow(shoot) :- !, retractall(a_hasArrow).
update_hasArrow(_).

update_hasGold(grab) :- !, asserta(a_hasGold).
update_hasGold(_).

update_perceiveBreeze(forward) :-
    a_hunterPosition(R),
    w_Pit(RP),
    isAdjacent(R,RP),
    asserta(a_perceiveBreeze(R)).
update_perceiveBreeze(init) :-
    a_hunterPosition(R),
    w_Pit(RP),
    isAdjacent(R,RP),
    asserta(a_perceiveBreeze(R)).
update_perceiveBreeze(_).

update_perceiveStench(forward) :-
    a_hunterPosition(R),
    w_Wumpus(RP),
    isAdjacent(R,RP),
    asserta(a_perceiveStench(R)).
update_perceiveStench(init) :-
    a_hunterPosition(R),
    w_Wumpus(RP),
    isAdjacent(R,RP),
    asserta(a_perceiveStench(R)).
update_perceiveStench(_).

update_perceiveGlitter(forward) :-
    a_hunterPosition(R),
    w_Gold(R),
    asserta(a_perceiveGlitter(R)).
update_perceiveGlitter(_).

%WORLD KNOWLEDGE
% Also work seamlessly by using information from past situations and
% perceptions to build on useful world knowledge.

%Agent remembers all rooms he has visited
a_visited(r(1,1)).

update_action(A) :- !,
    update_hunterDirection(A),
    update_hunterPosition(A),
    update_perceiveBreeze(A),
    update_perceiveStench(A),
    update_perceiveGlitter(A),
    update_hunterAlive(A),
    update_wumpusAlive(A),
    update_hasArrow(A),
    update_hasGold(A).

update_actions([]).
update_actions([H|T]) :- update_action(H), update_actions(T).

%WORLD EVALUATION
% These are extra tools given to the agent to evaluate world
% characteristics. These should be used by the heuristic to evaluate
% different possible actions and choose the best.

% A room is considered ok if there is no chance it has a pit or an alive wumpus
isOkRoom(R) :- \+possiblePit(R), (\+a_perceiveScream -> \+possibleWumpus(R);true).

% Evaluates possibility of pit in a certain room. Checks if all adjacent
% rooms that were visited had breezes
possiblePit(R) :- \+a_visited(R), getAdjacentRooms(R,LA), trimNotVisited(LA,LT), checkBreezeList(LT).
checkBreezeList([]).
checkBreezeList([H|T]) :- checkBreezeList(T), a_perceiveBreeze(H).

% One can only be certain of a pits position if there is a room with
% breeze where 3 adjacent rooms were visited and don't have a pit. The
% pit is in the fourth room certainly.
certainPit(RP) :-
    getAdjacentRooms(RP,LA),
    trimNotVisited(LA,LT),
    checkPitCertainty(RP,LT).

checkPitCertainty(_,[]) :- false.
checkPitCertainty(RP,[H|T]) :-
    a_perceiveBreeze(H),
    (
        (
        getAdjacentRooms(H,LA),
        trimVisited(LA,LT),
        trimWall(LT,LT2),
        LT2 = [RP]
        )
        ; checkPitCertainty(RP,T)
    ).

% Evaluates possibility of Wumpus in a certain room. Checks if all
% adjacent rooms that were visited had stench
possibleWumpus(R) :- certainWumpusExhaustive(R2), !, R = R2. %a certain Wumpus is also a possible Wumpus
possibleWumpus(R) :- \+a_visited(R), getAdjacentRooms(R,LA), trimNotVisited(LA,LT), checkStenchList(LT).
checkStenchList([]).
checkStenchList([H|T]) :- checkStenchList(T), a_perceiveStench(H).

% More easy than checking for pits, as we know there is only one
% Wumpus, one can mix and match adjacent rooms of two or more rooms with
% stench. If only one room that wasn't visited remains, the Wumpus must
% be there.
certainWumpus(RW) :-
    setof(R,a_perceiveStench(R),[H|T]), %H is going to be used as reference, and T will help
    getAdjacentRooms(H,LA),
    trimVisited(LA,LAT),
    trimWall(LAT,LATW),
    trimNotAdjacent(LATW,T,LT),
    LT = [RW]. %If only one room is reached, that is where the wumpus is

certainWumpusFromRef(RW,FL,[HR|_]) :- %HR is going to be used as reference
    getAdjacentRooms(HR,LA),
    trimVisited(LA,LAT),
    trimWall(LAT,LATW),
    delete(FL,HR,FLD), % don't check if HR is adjacent; it isn't
    trimNotAdjacent(LATW,FLD,LT),
    LT = [RW], %If only one room is reached, that is where the wumpus is
    asserta(a_wumpusPosition(RW)),
    !.
certainWumpusFromRef(RW,FL,[_|TR]) :-
    certainWumpusFromRef(RW,FL,TR).

certainWumpusExhaustive(RW) :-
    a_wumpusPosition(RW2),
    !,
    RW = RW2.
certainWumpusExhaustive(RW) :-
    setof(R,a_perceiveStench(R),FL),
    certainWumpusFromRef(RW,FL,FL).

% HEURISTIC
% Here is where you teach the intelligent agent different strategies
% to make decisions. At the end of the heuristic cycle, a major action
% must be returned. Possible major actions are:
%   move(R) - Move to a certain room. Should be used to explore or move
%       to strategic places.
%   grabGold - Should only be used if gold position is known. If not on
%       gold position, moves through known squares to go grab it.
%   shootWumpus - Should only be used if wumpus position is known. Does
%       the least amount of moves on known squares and shoots Wumpus.
%   exitCave - moves to cave entrance and climb out.
%   left, right, forward, grab, shoot, climb - does action without
%       checking anything.
%
% If you want to implement you own heuristic or strategy, do changes
% in the code below.
%
% This heuristic explores with the least amount of actions possible,
% shoots Wumpus as soon as he is certain and doesn't take risks on
% exploration. This will sometimes take a while!
heuristic(S,H) :-
    getAllExplorableRooms(L), %Get entire list of all rooms adjacent to rooms that were visited
%    format('All explorable rooms: ~w ~n', [L]),
    getBetterExplorableRoom(S,L,P,R),
%    format('Better explorable room: ~w ~n', R),
    heuristicPickAction(P,R,H).

heuristicPickAction(_,_,exitCave) :-  a_hasGold. %If hunter has gold, proceed to exit
heuristicPickAction(_,_,grabGold) :- \+a_hasGold, a_perceiveGlitter(_R). %If doesn't have gold but knows where it is, go get it
heuristicPickAction(_,_,shootWumpus) :- certainWumpusExhaustive(_RW),a_hasArrow,\+a_perceiveScream. %If is certain of where the Wumpus is, has arrow and Wumpus is alive, shoot him
heuristicPickAction(P,R,move(R)) :-  P < 10000, !. %Only move if best room to explore is not dangerous
heuristicPickAction(_,_,exitCave).  %If no rooms to explore, exit cave

getBetterExplorableRoom(_S,[],5000,_R) :- !. %Only run ranking of rooms if there are rooms to rank
getBetterExplorableRoom(S,L,P,R) :-
    rankRooms(L,S,RL),
    sort(RL,SRL),
    [rr(P,R)|_] = SRL.

% Ranks rooms by number of actions to explore and danger levels
rankRooms([],_,[]).
rankRooms([H|T],S,[rr(Total,H)|LT]) :-
    rankRooms(T,S,LT),
    %Count actions
    doMove(H,ST,S),
    countActions(ST,S,NActions),
    %Check breeze and stench
    (isOkRoom(H) -> DangerPoints = 0; DangerPoints = 100),
    %Check certain Pit and Wumpus
    (certainPit(H) -> CertainPitPoints = 1000; CertainPitPoints = 0),
    ((\+a_perceiveScream, certainWumpusExhaustive(H)) -> CertainWumpusPoints = 1000; CertainWumpusPoints = 0),
    Total is NActions + DangerPoints + CertainPitPoints + CertainWumpusPoints. %Saves rank for each room.

%PLANNING
% The following clauses should be used for planning of actions. Planning
% will perform a Breadth First Search from a certain situation using
% actions to reach a desired goal. When performing planning for
% sequences of more than 10 actions this can take a long while.

% Preconditions for primitive actions. Define whether an action can be
% taken at each situation.
poss(forward,S) :- %Allow planning only on visited and ok rooms.
    hunter(R,D,S),
    getForwardRoom(R,D,RF),
    isOkRoom(RF).
poss(left,s0).
poss(left,do(A,_S)) :- \+A = right. %Limit redundant turning
poss(right,s0).
poss(right,do(A,_S)) :- \+A = left. %Limit redundant turning

%Legality axioms - Makes certain that a situation is possible and legal
%legal(S,S0) reads: If S0 is legal, return whether S is legal
legal(S,S). %If S is legal, S is legal
legal(do(A,S),S0):-
    maxNumberOfActionsPlanned(Max), %Get maximum allowed number of actions
    legal(S,S0), %Tries to find legal actions, starting from provided situation S0
    countActions(S,S0,N), %Count number of actions from S0 to S
    (N > Max -> (!, write('REACHED MAX NUMBER OF ACTIONS PLANNED'),false) ; true), %If too many actions are being taken, probably there is no solution, hence return false
    poss(A,S). %Check which actions are allowed at S

% Movement planner - The last forward action is forced, even if it
% doesn't result in the hunter's movement. That must be done because if
% the hunter hits a wall it won't know it hasn't moved until it receive
% a bump as a perception.
% doMove(R,S,S0) returns a plan of a sequence of movement actions that
% make the hunter in situation S0 move to R. S is returned as the
% resulting situation.
doMove(Rm, S0, S0) :- hunter(Rm,_,S0). %Moving to where the hunter is returns no actions
doMove(Rm, do(forward,S), S0) :- legal(S,S0),hunter(R,D,S),isAdjacent(R,Rm),isFacing(R,D,Rm),!. %Reads: Which is a situation S supposing S0 is legal, where the hunter is at R?

doFace(Rm, S, S0) :- legal(S,S0),hunter(R,D,S),isFacing(R,D,Rm),!. %Similar to doMove, but only faces de target

%ACTUATOR
% After the heuristic defines a major action, this clause will convert
% that action to a situation with planning. Passing this situation to
% the next loops counts as acting.
doActions(move(R),S,S0) :- doMove(R,S,S0). %Move
doActions(grabGold,do(grab,SI),S0) :- a_perceiveGlitter(R), doMove(R,SI,S0). %Move and then grab
doActions(shootWumpus,do(shoot, SI),S0) :- certainWumpusExhaustive(RW), doFace(RW,SI,S0). %Face Wumpus and shoot
doActions(exitCave,do(climb, SI),S0) :- hunter(R0,_,s0), doMove(R0,SI,S0). %Moves to entry and climbs
doActions(climb,do(climb,S0),S0). %Climb
doActions(forward,do(forward, S0),S0).
doActions(left,do(left, S0),S0).
doActions(right,do(right, S0),S0).
doActions(grab,do(grab, S0),S0).
doActions(shoot,do(shoot, S0),S0).


%INTELLIGENT AGENT LOOP
% An entire Agent Loop consists of perceptions, Gathering World
% Knowledge, Heuristic (Deciding actions), Planning and acting. The
% following clauses will run the loops while printing relevant
% information so that we can watch our little AI moving. Each loop
% consists of a few smaller actions that are planned, but only one major
% heuristic action.
% In this version of the program, one should use the runManyMaps(N0,NF)
% clause to run a bunch of maps in sequence.
runManyMaps(N0,NF) :- %Runs map N0 until NF inclusive in sequence.
    consult('C:/Users/fudal/OneDrive/Dokumenty/GitHub/inteligencja-obliczeniowa-projekty/2/worldBuilder.pl'), %This file has information for different maps
    make, %Reset files if changed
    runInSequence(N0,NF). %Runs many maps in sequence

run :-
    consult('C:/Users/fudal/OneDrive/Dokumenty/GitHub/inteligencja-obliczeniowa-projekty/2/worldBuilder.pl'), %This file has information for different maps
    run(1). %Ruins AIMA Map


runInSequence(N0,NF) :- %This loops through different maps and runs agent in each one
    run(N0),
    N1 is N0+1,
    (N1 =< NF -> runInSequence(N1,NF) ; true). %Run next map if not done.

run(N) :-
    recreateWorld(N),
    resetWorld,
%    format('~n~n~n   Playing on world ~d ~n~n~n', N),
    callTime(runloop(0,s0)).

% This clause is called before the actual loop to check if maximum
% number of moves has been reached (Stops if its taking too long)
runloop(T,_) :-
    maxNumberOfMoves(Max),
    T >= Max,
    write('Reached max number of moves'), !, false.

%Main loop.
runloop(T,S0) :-
    %Gets current hunter position and prints.
    hunter(r(X,Y),D,S0),
    update_actions([init]),
%    format('Hunter at [~d,~d, ~w], ', [X,Y,D]),!,
%    printState(S0),

    %Get action from heuristic (Strategy) in this situation
    heuristic(S0,H),
%    format('wants to do ~w, ', [H]), %Prints desired action

    %Calls actuators which use planning to do Actions
    doActions(H,S,S0),
%    write('does '),
    planToActionList(S,S0,[],L),
%    printList(L), %Prints list of all smaller actions that were done.
    update_actions(L),

%    ((certainWumpusExhaustive(RW),\+a_perceiveScream) -> format('I am certain Wumpus is at ~w',RW);true), %Prints Wumpus position if certain
%    format('~n'),

    NT is T+1, %Set new timestep

    %The following are needed to check if hunter climbed out of the cave
    do(A,_) = S, %Get last action done
    hunter(RN,_,S), %Get hunters position now
    hunter(R0,_,s0), %Get Cave entry

    %If hunter climbed out or died, endloop. If not, run next loop.
    (   ((A = climb, RN = R0)  ; \+a_hunterAlive) -> endLoop(S,A)
    ;   (!,runloop(NT,S))
    ),!.

% After Ending the loop, game is over, print everything that is
% interesting to file.
endLoop(S,A) :-
    countActions(S,s0,N),
%    format('~n~n   '),
%    (a_hasGold -> write('Got the gold'); write('Couldnt find gold')),
%    ( \+a_hunterAlive -> write(' and died') ; write(' and left the cave')),
    format('actions ~d',N),
    %Scoring
    ((a_hasGold,A = climb) -> GoldPoints is 1000 ; GoldPoints is 0),
    (\+a_hunterAlive -> DeathPoints is -1000 ; DeathPoints is 0),
    (a_hasArrow ->ArrowPoints is 0 ; ArrowPoints is -10),
    Score is GoldPoints + DeathPoints + ArrowPoints - N,
    format(' Points ~d ',Score).

%HELPERS
%These are helper functions that make the programming above easier
add(E,L,[E|L]). %Adds element to list

trimNotVisited([],[]). %Removes rooms that weren't visited from list of rooms
trimNotVisited([H|T],LT) :- trimNotVisited(T,L), (a_visited(H) -> append([H],L,LT); LT = L).
trimVisited([],[]). %Removes rooms that were visited from list of rooms
trimVisited([H|T],LT) :- trimVisited(T,L), (a_visited(H) -> LT = L; append([H],L,LT)).
trimWall([],[]). %Removes rooms that have been confirmed as walls from list of rooms
trimWall([H|T],LT) :- trimWall(T,L), (w_Wall(H) -> LT = L; append([H],L,LT)).
trimNotAdjacent([],_,[]). %used as trimNotAdjacent(L,T,LT)
trimNotAdjacent(_,[],[]). %Removes rooms from List L that are no adjacent to any room in list T
trimNotAdjacent([LAH|LAT],[TH|TT],LT) :-
    trimNotAdjacent([LAH],TT,LT1),
    trimNotAdjacent(LAT,[TH|TT],LT2),
    append(LT1,LT2,LT3),
    (isAdjacent(LAH,TH) -> append([LAH],LT3,LT) ; LT = LT3).

%Converts plan (Actions from one situation to another) to Action list
planToActionList(S,S,ACC,ACC).
planToActionList(do(A,S1),S0,ACC,X) :- planToActionList(S1,S0,[A|ACC],X).

%Prints List
printList([]).
printList([A|B]) :-
    format('~w, ', A),
    printList(B).

%Returns room in front of another in a certain direction
getForwardRoom(r(X0,Y0),n,r(XN,YN)) :- XN is X0, YN is Y0+1.
getForwardRoom(r(X0,Y0),e,r(XN,YN)) :- XN is X0+1, YN is Y0.
getForwardRoom(r(X0,Y0),s,r(XN,YN)) :- XN is X0, YN is Y0-1.
getForwardRoom(r(X0,Y0),w,r(XN,YN)) :- XN is X0-1, YN is Y0.

%Checks if one room is adjacent to another room
isAdjacent(r(X,Y),r(XT,YT)) :- (X =:= XT, Y =:= YT+1).
isAdjacent(r(X,Y),r(XT,YT)) :- (X =:= XT, Y =:= YT-1).
isAdjacent(r(X,Y),r(XT,YT)) :- (X =:= XT+1, Y =:= YT).
isAdjacent(r(X,Y),r(XT,YT)) :- (X =:= XT-1, Y =:= YT).

%Checks if a hunter in room R, looking to Direction D is facing room RT
isFacing(r(X,Y),n,r(XT,YT)) :- X =:= XT, YT > Y.
isFacing(r(X,Y),s,r(XT,YT)) :- X =:= XT, YT < Y.
isFacing(r(X,Y),e,r(XT,YT)) :- Y =:= YT, XT > X.
isFacing(r(X,Y),w,r(XT,YT)) :- Y =:= YT, XT < X.

%Returns list of all adjacent rooms
getAdjacentRooms(r(X,Y),L) :-
    XL is X-1,
    XR is X+1,
    YD is Y-1,
    YU is Y+1,
    append([r(XL,Y), r(XR,Y), r(X,YU), r(X,YD)],[],L).

% The following functions are used to get a list of explorable rooms.
% Those are rooms adjacent to rooms that were already visited. All rooms
% on the border of what has been explored. In a certain situation S a
% list L is returned with all possible rooms.

processExplorableRooms([], []).
processExplorableRooms([H|T], L) :-
    processExplorableRooms(T, LT),
    getAdjacentRooms(H,LA),
    appendWithExplorableCheck(LA,LT,L).
getAllExplorableRooms(L) :-
    setof(R,a_visited(R),LV),
    processExplorableRooms(LV,L).

appendWithExplorableCheck([],L2,L2).
appendWithExplorableCheck([H|T],L2,L) :-
    appendWithExplorableCheck(T,L2,LT),
    (   isExplorable(H,LT) -> L = [H|LT] ; L = LT).

isExplorable(R,L) :- \+member(R,L), \+w_Wall(R), \+a_visited(R).

%Counts number of actions between two situations
countActions(s0,s0,0).
countActions(S,S,0).
countActions(do(_A,S),S0,N) :- %Count number of actions between two situations
    countActions(S,S0,N0),
    N is N0+1.

callTime(G,T) :- %Returns Call Time
    statistics(runtime,[T0|_]),
    G,
    statistics(runtime,[T1|_]),
    T is T1 - T0.

callTime(G) :- %Prints Call Time
    callTime(G,T),
    format('time ~d ~n',T).


printWorldBlock(B) :- w_Wall(B), !, format('#').
printWorldBlock(B) :- w_Gold(B), !, format('G').
printWorldBlock(B) :- w_Pit(B), !, format('P').
printWorldBlock(B) :- w_Wumpus(B), !, format('W').
printWorldBlock(r(1,1)) :- !, format('e').
printWorldBlock(_) :- format(' ').

printWorldLine(K) :-
    printWorldBlock(r(0,K)),
    printWorldBlock(r(1,K)),
    printWorldBlock(r(2,K)),
    printWorldBlock(r(3,K)),
    printWorldBlock(r(4,K)),
    printWorldBlock(r(5,K)).

printWorld(N) :- 
    recreateWorld(N),
    format('~n~n~n   Displaying world ~d ~n~n~n', N),
    printWorldLine(5),
    format('~n'),
    printWorldLine(4),
    format('~n'),
    printWorldLine(3),
    format('~n'),
    printWorldLine(2),
    format('~n'),
    printWorldLine(1),
    format('~n'),
    printWorldLine(0),
    format('~n').



printGoldState(B) :- a_perceiveGlitter(B), !, format('g').
printGoldState(B) :- a_visited(B), !, format('v').
printGoldState(_B) :- !, format('?').

printPitState(_S, B) :- certainPit(B), !, format('P').
printPitState(_S, B) :- possiblePit(B), !, format('p').
printPitState(_S, B) :- a_perceiveBreeze(B), !, format('b').
printPitState(_S, _B) :- !, format(' ').

printWumpusState(B) :- certainWumpusExhaustive(B), a_wumpusAlive, !, format('U').
printWumpusState(B) :- certainWumpusExhaustive(B), \+a_wumpusAlive, !, format('F').
printWumpusState(B) :- possibleWumpus(B), !, format('u').
printWumpusState(B) :- a_perceiveStench(B), !, format('|').
printWumpusState(_) :- !, format(' ').


printStateBlock(_S, B) :- w_Wall(B), !, format('####').
printStateBlock(S, B) :-
    (hunter(B, D, S) -> format('~a', D); format(' ')),
    printGoldState(B),
    printPitState(S, B),
    printWumpusState(B).


printStateLine(S, K) :-
    printStateBlock(S, r(0,K)),
    printStateBlock(S, r(1,K)),
    printStateBlock(S, r(2,K)),
    printStateBlock(S, r(3,K)),
    printStateBlock(S, r(4,K)),
    printStateBlock(S, r(5,K)).


printState(S) :-
    format('~n~n   Displaying current knowledge state; [hunter?, gold?, pit?, wumpus?] ~n~n'),
    printStateLine(S, 5),
    format('~n'),
    printStateLine(S, 4),
    format('~n'),
    printStateLine(S, 3),
    format('~n'),
    printStateLine(S, 2),
    format('~n'),
    printStateLine(S, 1),
    format('~n'),
    printStateLine(S, 0),
    format('~n~n').
