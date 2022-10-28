(define (domain igibson)
    (:requirements :strips :adl)
    (:types
        human phyobj - object
    )
    (:predicates
        (human-at ?h - human ?o - phyobj)
        (is-working ?h - human)
        (is-waiting ?h - human)
        (holding ?h - human ?o - phyobj)
        (hand-empty ?h - human)

        (movable ?o - phyobj)
        (receptacle ?o - phyobj)

        (inside ?o - phyobj ?o - phyobj)
        (ontop ?o - phyobj ?o - phyobj)
        (has-inside ?o - phyobj)
        (has-ontop ?o - phyobj)

        (has-size ?o - phyobj)
        (has-color ?o - phyobj)
        (size-large ?o - phyobj)
        (size-small ?o - phyobj)
        (color-red ?o - phyobj)
        (color-blue ?o - phyobj)
        (color-green ?o - phyobj)

		(type-floor ?o - phyobj)
		(type-countertop ?o - phyobj)
		(type-sofa ?o - phyobj)
		(type-bed ?o - phyobj)
		(type-stove ?o - phyobj)
		(type-table ?o - phyobj)
		(type-shelf ?o - phyobj)
		(type-toilet ?o - phyobj)
		(type-cabinet ?o - phyobj)
		(type-bathtub ?o - phyobj)
		(type-microwave ?o - phyobj)
		(type-oven ?o - phyobj)
		(type-dishwasher ?o - phyobj)
		(type-refrigerator ?o - phyobj)
		(type-sink ?o - phyobj)
		(type-pool ?o - phyobj)
		(type-highchair ?o - phyobj)
		(type-chair ?o - phyobj)
		(type-seat ?o - phyobj)
		(type-bottle ?o - phyobj)
		(type-jar ?o - phyobj)
		(type-kettle ?o - phyobj)
		(type-caldron ?o - phyobj)
		(type-bowl ?o - phyobj)
		(type-mug ?o - phyobj)
		(type-plate ?o - phyobj)
		(type-dish ?o - phyobj)
		(type-cup ?o - phyobj)
		(type-saucepan ?o - phyobj)
		(type-pan ?o - phyobj)
		(type-teapot ?o - phyobj)
		(type-blender ?o - phyobj)
		(type-casserole ?o - phyobj)
		(type-duffel_bag ?o - phyobj)
		(type-sack ?o - phyobj)
		(type-backpack ?o - phyobj)
		(type-briefcase ?o - phyobj)
		(type-bucket ?o - phyobj)
		(type-tray ?o - phyobj)
		(type-basket ?o - phyobj)
		(type-box ?o - phyobj)
		(type-package ?o - phyobj)
		(type-ashcan ?o - phyobj)
		(type-xmas_stocking ?o - phyobj)
		(type-xmas_tree ?o - phyobj)
		(type-apple ?o - phyobj)
		(type-banana ?o - phyobj)
		(type-melon ?o - phyobj)
		(type-grape ?o - phyobj)
		(type-lemon ?o - phyobj)
		(type-orange ?o - phyobj)
		(type-peach ?o - phyobj)
		(type-strawberry ?o - phyobj)
		(type-raspberry ?o - phyobj)
		(type-date ?o - phyobj)
		(type-olive ?o - phyobj)
		(type-chestnut ?o - phyobj)
		(type-carrot ?o - phyobj)
		(type-radish ?o - phyobj)
		(type-tomato ?o - phyobj)
		(type-broccoli ?o - phyobj)
		(type-mushroom ?o - phyobj)
		(type-onion ?o - phyobj)
		(type-lettuce ?o - phyobj)
		(type-pumpkin ?o - phyobj)
		(type-pop ?o - phyobj)
		(type-beer ?o - phyobj)
		(type-juice ?o - phyobj)
		(type-water ?o - phyobj)
		(type-milk ?o - phyobj)
		(type-beef ?o - phyobj)
		(type-chicken ?o - phyobj)
		(type-pork ?o - phyobj)
		(type-fish ?o - phyobj)
		(type-egg ?o - phyobj)
		(type-catsup ?o - phyobj)
		(type-sauce ?o - phyobj)
		(type-parsley ?o - phyobj)
		(type-tea_bag ?o - phyobj)
		(type-sugar ?o - phyobj)
		(type-vegetable_oil ?o - phyobj)
		(type-cracker ?o - phyobj)
		(type-bread ?o - phyobj)
		(type-cookie ?o - phyobj)
		(type-cake ?o - phyobj)
		(type-chip ?o - phyobj)
		(type-hamburger ?o - phyobj)
		(type-sandwich ?o - phyobj)
		(type-candy ?o - phyobj)
		(type-oatmeal ?o - phyobj)
		(type-sushi ?o - phyobj)
		(type-salad ?o - phyobj)
		(type-soup ?o - phyobj)
		(type-pasta ?o - phyobj)
		(type-carving_knife ?o - phyobj)
		(type-hammer ?o - phyobj)
		(type-screwdriver ?o - phyobj)
		(type-scraper ?o - phyobj)
		(type-saw ?o - phyobj)
		(type-printer ?o - phyobj)
		(type-scanner ?o - phyobj)
		(type-facsimile ?o - phyobj)
		(type-modem ?o - phyobj)
		(type-calculator ?o - phyobj)
		(type-headset ?o - phyobj)
		(type-earphone ?o - phyobj)
		(type-mouse ?o - phyobj)
		(type-alarm ?o - phyobj)
		(type-toothbrush ?o - phyobj)
		(type-perfume ?o - phyobj)
		(type-makeup ?o - phyobj)
		(type-highlighter ?o - phyobj)
		(type-marker ?o - phyobj)
		(type-pen ?o - phyobj)
		(type-pencil ?o - phyobj)
		(type-dishtowel ?o - phyobj)
		(type-hand_towel ?o - phyobj)
		(type-rag ?o - phyobj)
		(type-scrub_brush ?o - phyobj)
		(type-broom ?o - phyobj)
		(type-vacuum ?o - phyobj)
		(type-soap ?o - phyobj)
		(type-shampoo ?o - phyobj)
		(type-detergent ?o - phyobj)
		(type-toothpaste ?o - phyobj)
		(type-fork ?o - phyobj)
		(type-spoon ?o - phyobj)
		(type-knife ?o - phyobj)
		(type-lamp ?o - phyobj)
		(type-candle ?o - phyobj)
		(type-necklace ?o - phyobj)
		(type-bracelet ?o - phyobj)
		(type-jewelery ?o - phyobj)
		(type-bow ?o - phyobj)
		(type-wreath ?o - phyobj)
		(type-ribbon ?o - phyobj)
		(type-hardback ?o - phyobj)
		(type-notebook ?o - phyobj)
		(type-book ?o - phyobj)
		(type-newspaper ?o - phyobj)
		(type-painting ?o - phyobj)
		(type-pad ?o - phyobj)
		(type-document ?o - phyobj)
		(type-gym_shoe ?o - phyobj)
		(type-sandal ?o - phyobj)
		(type-shoe ?o - phyobj)
		(type-sock ?o - phyobj)
		(type-hat ?o - phyobj)
		(type-sunglass ?o - phyobj)
		(type-shirt ?o - phyobj)
		(type-sweater ?o - phyobj)
		(type-underwear ?o - phyobj)
		(type-apparel ?o - phyobj)
		(type-tile ?o - phyobj)
		(type-plywood ?o - phyobj)
		(type-cube ?o - phyobj)
		(type-ball ?o - phyobj)

        (open ?o - phyobj)
        (cooked ?o - phyobj)
        (dusty ?o - phyobj)
        (frozen ?o - phyobj)
        (stained ?o - phyobj)
        (sliced ?o - phyobj)
        (soaked ?o - phyobj)
        (toggled ?o - phyobj)

        (openable ?o - phyobj)
        (cookable ?o - phyobj)
        (dustyable ?o - phyobj)
        (freezable ?o - phyobj)
        (stainable ?o - phyobj)
        (sliceable ?o - phyobj)
        (soakable ?o - phyobj)
        (toggleable ?o - phyobj)

        (valid-clean-pair ?o - phyobj ?o - phyobj)
        ;(multiple_pickable ?o - phyobj)
        ;(picked_once ?o - phyobj)
    )


    (:action human-move
        :parameters (?h - human ?from - phyobj ?to - phyobj)
        :precondition (and (is-working ?h) (human-at ?h ?from) (not (movable ?from)) (not (movable ?to)))
        :effect (and (human-at ?h ?to) (not (human-at ?h ?from)))
    )

    (:action human-pick-up-at-location
        :parameters (?h - human ?obj - phyobj ?loc - phyobj)
        :precondition (and (is-working ?h) (human-at ?h ?loc) (hand-empty ?h)
                        (or (and (inside ?obj ?loc) (or (not (openable ?loc)) (open ?loc))) (ontop ?obj ?loc))
                        (not (movable ?loc)) (movable ?obj) (receptacle ?loc)
                        ;(not (picked_once ?obj))
                        )
        :effect (and (not (hand-empty ?h)) (not (inside ?obj ?loc)) (not (ontop ?obj ?loc)) (holding ?h ?obj)
                ;(picked_once ?obj)
                )
    )

    (:action human-pick-up-from-receptacle-at-location
        :parameters (?h - human ?obj - phyobj ?rec - phyobj ?loc - phyobj)
        :precondition (and (is-working ?h) (human-at ?h ?loc) (hand-empty ?h)
                        (or (and (inside ?obj ?rec) (or (not (openable ?rec)) (open ?rec))) (ontop ?obj ?rec))
                        (or (and (inside ?rec ?loc) (or (not (openable ?loc)) (open ?loc))) (ontop ?rec ?loc))
                        (not (movable ?loc)) (movable ?rec) (movable ?obj)
                        (receptacle ?loc) (receptacle ?rec) (not (receptacle ?obj))
                        ;(not (picked_once ?obj))
                        )
        :effect (and (not (hand-empty ?h)) (not (inside ?obj ?rec)) (not (ontop ?obj ?rec)) (holding ?h ?obj)
                ;(picked_once ?obj)
                )
    )

    (:action human-put-inside-location
        :parameters (?h - human ?obj - phyobj ?loc - phyobj)
        :precondition (and (is-working ?h) (human-at ?h ?loc) (holding ?h ?obj)
                        (not (movable ?loc)) (movable ?obj) (receptacle ?loc)
                        (has-inside ?loc) (or (not (openable ?loc)) (open ?loc)))
        :effect (and (not (holding ?h ?obj)) (inside ?obj ?loc) (hand-empty ?h))
    )

    (:action human-put-ontop-location
        :parameters (?h - human ?obj - phyobj ?loc - phyobj)
        :precondition (and (is-working ?h) (human-at ?h ?loc) (holding ?h ?obj)
                        (not (movable ?loc)) (movable ?obj) (receptacle ?loc)
                        (has-ontop ?loc))
        :effect (and (not (holding ?h ?obj)) (ontop ?obj ?loc) (hand-empty ?h))
    )

    (:action human-put-inside-receptacle-at-location
        :parameters (?h - human ?obj - phyobj ?rec - phyobj ?loc - phyobj)
        :precondition (and (is-working ?h) (human-at ?h ?loc) (holding ?h ?obj)
                        (or (and (inside ?rec ?loc) (or (not (openable ?loc)) (open ?loc))) (ontop ?rec ?loc))
                        (has-inside ?rec) (or (not (openable ?rec)) (open ?rec))
                        (not (movable ?loc)) (movable ?rec) (movable ?obj)
                        (receptacle ?loc) (receptacle ?rec) (not (receptacle ?obj)))
        :effect (and (not (holding ?h ?obj)) (inside ?obj ?rec) (hand-empty ?h))
    )

    (:action human-put-ontop-receptacle-at-location
        :parameters (?h - human ?obj - phyobj ?rec - phyobj ?loc - phyobj)
        :precondition (and (is-working ?h) (human-at ?h ?loc) (holding ?h ?obj)
                        (or (and (inside ?rec ?loc) (or (not (openable ?loc)) (open ?loc))) (ontop ?rec ?loc))
                        (has-ontop ?rec)
                        (not (movable ?loc)) (movable ?rec) (movable ?obj)
                        (receptacle ?loc) (receptacle ?rec) (not (receptacle ?obj)))
        :effect (and (not (holding ?h ?obj)) (ontop ?obj ?rec) (hand-empty ?h))
    )

    (:action human-open-location
        :parameters (?h - human ?loc - phyobj)
        :precondition (and (is-working ?h) (openable ?loc) (human-at ?h ?loc) (not (open ?loc)))
        :effect (and (open ?loc))
    )

    (:action human-close-location
        :parameters (?h - human ?loc - phyobj)
        :precondition (and (is-working ?h) (openable ?loc) (human-at ?h ?loc) (open ?loc))
        :effect (and (not (open ?loc)))
    )

    (:action human-open-receptacle-at-location
        :parameters (?h - human ?rec - phyobj ?loc - phyobj)
        :precondition (and (is-working ?h) (openable ?rec) (human-at ?h ?loc) (not (open ?rec))
                        (or (and (inside ?rec ?loc) (or (not (openable ?loc)) (open ?loc))) (ontop ?rec ?loc)))
        :effect (and (open ?rec))
    )

    (:action human-close-receptacle-at-location
        :parameters (?h - human ?rec - phyobj ?loc - phyobj)
        :precondition (and (is-working ?h) (openable ?rec) (human-at ?h ?loc) (open ?rec)
                        (or (and (inside ?rec ?loc) (or (not (openable ?loc)) (open ?loc))) (ontop ?rec ?loc)))
        :effect (and (not (open ?rec)))
    )

    (:action human-toggle-on-location
        :parameters (?h - human ?loc - phyobj)
        :precondition (and (is-working ?h) (toggleable ?loc) (human-at ?h ?loc) (not (toggled ?loc)))
        :effect (and (toggled ?loc))
    )

    (:action human-toggle-off-location
        :parameters (?h - human ?loc - phyobj)
        :precondition (and (is-working ?h) (toggleable ?loc) (human-at ?h ?loc) (toggled ?loc))
        :effect (and (not (toggled ?loc)))
    )

    (:action human-toggle-on-movable-at-location
        :parameters (?h - human ?obj - phyobj ?loc - phyobj)
        :precondition (and (is-working ?h) (toggleable ?obj) (human-at ?h ?loc) (not (toggled ?obj))
                        (or (and (inside ?obj ?loc) (or (not (openable ?loc)) (open ?loc))) (ontop ?obj ?loc)))
        :effect (and (toggled ?obj))
    )

    (:action human-toggle-off-movable-at-location
        :parameters (?h - human ?obj - phyobj ?loc - phyobj)
        :precondition (and (is-working ?h) (toggleable ?obj) (human-at ?h ?loc) (toggled ?obj)
                        (or (and (inside ?obj ?loc) (or (not (openable ?loc)) (open ?loc))) (ontop ?obj ?loc)))
        :effect (and (not (toggled ?obj)))
    )

    (:action human-heat-obj
        :parameters (?h - human ?obj - phyobj ?loc - phyobj)
        :precondition (and (is-working ?h) (hand-empty ?h) (human-at ?h ?loc) (cookable ?obj)
                        (or (type-microwave ?loc) (type-stove ?loc) (type-oven ?loc)) (toggled ?loc)
                        (or (inside ?obj ?loc) (ontop ?obj ?loc))
                        (not (receptacle ?obj)) (not (movable ?loc)))
        :effect (and (cooked ?obj) (not (toggled ?loc)) (not (frozen ?obj)))
    )

    (:action human-cool-obj
        :parameters (?h - human ?obj - phyobj ?loc - phyobj)
        :precondition (and (is-working ?h) (hand-empty ?h) (human-at ?h ?loc) (inside ?obj ?loc) (freezable ?obj)
                        (or (type-refrigerator ?loc)) (toggled ?loc) (not (receptacle ?obj)) (not (movable ?loc)))
        :effect (and (frozen ?obj) (not (cooked ?obj)))
    )

    (:action human-slice-obj
        :parameters (?h - human ?obj - phyobj ?tool - phyobj ?loc - phyobj)
        :precondition (and (is-working ?h) (holding ?h ?tool) (type-knife ?tool) (sliceable ?obj)
                        (human-at ?h ?loc) (ontop ?obj ?loc) (not (receptacle ?obj)) (not (movable ?loc)))
        :effect (and (sliced ?obj))
    )

    (:action human-soak-obj
        :parameters (?h - human ?obj - phyobj ?loc - phyobj)
        :precondition (and (is-working ?h) (human-at ?h ?loc) (inside ?obj ?loc) (soakable ?obj)
                        (type-sink ?loc) (toggled ?loc) (not (receptacle ?obj)) (not (movable ?loc)))
        :effect (and (soaked ?obj) (not (toggled ?loc)))
    )

    (:action human-clean-obj-at-location
        :parameters (?h - human ?obj - phyobj ?tool - phyobj ?loc - phyobj)
        :precondition (and (is-working ?h) (holding ?h ?tool) (or (dustyable ?obj) (stainable ?obj))
                        (human-at ?h ?loc) (movable ?obj) (not (movable ?loc))
                        (or (and (inside ?obj ?loc) (or (not (openable ?loc)) (open ?loc))) (ontop ?obj ?loc))
                        (or (not (soakable ?tool)) (soaked ?tool)) (valid-clean-pair ?obj ?tool))
        :effect (and (not (dusty ?obj)) (not (stained ?obj)))
    )

    (:action human-clean-location
        :parameters (?h - human ?tool - phyobj ?loc - phyobj)
        :precondition (and (is-working ?h) (holding ?h ?tool) (or (dustyable ?loc) (stainable ?loc))
                        (human-at ?h ?loc) (not (movable ?loc))
                        (or (not (soakable ?tool)) (soaked ?tool)) (valid-clean-pair ?loc ?tool))
        :effect (and (not (dusty ?loc)) (not (stained ?loc)))
    )

    ;;; unified actions for robots
    (:action robot-move-obj-to-human
        :parameters (?h - human ?obj - phyobj ?from - phyobj)
        :precondition (and (is-waiting ?h) (hand-empty ?h) (or (inside ?obj ?from) (ontop ?obj ?from))
                        (movable ?obj) (receptacle ?from))
        :effect (and (not (inside ?obj ?from)) (not (ontop ?obj ?from))
                        (not (hand-empty ?h)) (holding ?h ?obj) (is-working ?h) (not (is-waiting ?h)))
    )

    (:action robot-move-obj-from-rec-into-rec
        :parameters (?h - human ?obj - phyobj ?from - phyobj ?to - phyobj)
        :precondition (and (is-waiting ?h) (hand-empty ?h) (or (inside ?obj ?from) (ontop ?obj ?from))
                        (movable ?obj) (receptacle ?from) (receptacle ?to) (has-inside ?to)
                        (or (not (movable ?to)) (not (receptacle ?obj))))
        :effect (and (not (inside ?obj ?from)) (not (ontop ?obj ?from)) (inside ?obj ?to)
                        (is-working ?h) (not (is-waiting ?h)))
    )

    (:action robot-move-obj-from-rec-onto-rec
        :parameters (?h - human ?obj - phyobj ?from - phyobj ?to - phyobj)
        :precondition (and (is-waiting ?h) (hand-empty ?h) (or (inside ?obj ?from) (ontop ?obj ?from))
                        (movable ?obj) (receptacle ?from) (receptacle ?to) (has-ontop ?to)
                        (or (not (movable ?to)) (not (receptacle ?obj))))
        :effect (and (not (inside ?obj ?from)) (not (ontop ?obj ?from)) (ontop ?obj ?to)
                        (is-working ?h) (not (is-waiting ?h)))
    )
    (:action robot-toggle-on
        :parameters (?h - human ?obj - phyobj)
        :precondition (and (is-waiting ?h) (hand-empty ?h) (toggleable ?obj) (not (toggled ?obj)))
        :effect (and (toggled ?obj) (is-working ?h) (not (is-waiting ?h)))
    )

    (:action robot-heat-obj
        :parameters (?h - human ?obj - phyobj)
        :precondition (and (is-waiting ?h) (hand-empty ?h) (cookable ?obj))
        :effect (and (cooked ?obj) (not (frozen ?obj)) (is-working ?h) (not (is-waiting ?h)))
    )

    (:action robot-cool-obj
        :parameters (?h - human ?obj - phyobj)
        :precondition (and (is-waiting ?h) (hand-empty ?h) (freezable ?obj))
        :effect (and (frozen ?obj) (not (cooked ?obj)) (is-working ?h) (not (is-waiting ?h)))
    )

    (:action robot-slice-obj
        :parameters (?h - human ?obj - phyobj)
        :precondition (and (is-waiting ?h) (hand-empty ?h) (sliceable ?obj))
        :effect (and (sliced ?obj) (is-working ?h) (not (is-waiting ?h)))
    )

    (:action robot-soak-obj
        :parameters (?h - human ?obj - phyobj)
        :precondition (and (is-waiting ?h) (hand-empty ?h) (soakable ?obj))
        :effect (and (soaked ?obj) (is-working ?h) (not (is-waiting ?h)))
    )

    (:action robot-clean-obj
        :parameters (?h - human ?obj - phyobj)
        :precondition (and (is-waiting ?h) (hand-empty ?h) (or (stainable ?obj) (dustyable ?obj)))
        :effect (and (not (stained ?obj)) (not (dusty ?obj)) (is-working ?h) (not (is-waiting ?h)))
    )

)