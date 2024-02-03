---
title: Ludum Dare 47
header-img: /images/ludum47screenshot.webp
header-img-alt: A screenshot of my submission to Ludum Dare 47
tags: game dev, art, programming, project
---

[Digital Friendship Adventure, aka Delightful Feline Acquaintance, aka Dependable Female Attention, aka DFA](https://ksadov.github.io/ludumdare47/), is a minimalist dating simulator implemented as a deterministic finite automaton. Deterministic finite automata (henceforth denoted DFAs) are simple abstract models of computers. DFAs are defined by 5 elements:

* A finite set of states
* A finite set of input symbols
* A transition function which takes a state and a symbol as inputs, and outputs another state
* A starting state
* A set of accept states

In the game, the finite set of states is the set of the catgirl’s expressions. The input signals are the buttons at the bottom: `:)`, `:(`, `:D`, `>:(`, and `;)`. The transition function takes the catgirl’s current expression into account to produce an expression in response to the player’s input. The starting state is the catgirl’s neutral expression.

The natural accept states in a game implemented with a DFA would be some kind of win states. DFA doesn’t have an explicitly designated win state, but based on what little I knew about dating sims, I figured that the most desirable state would be the most lovestruck one:

![](https://raw.githubusercontent.com/ksadov/ludumdare47/main/src/hearteyes2.png)

so I designed it as kind of an attractor state: repeated positive interactions (`:)`, `:(`, `:D`, and `;)`) lead into and preserve it. But I learned from player feedback that people particularly liked this state:

![](https://raw.githubusercontent.com/ksadov/ludumdare47/main/src/bigsmile.png)

which is only reachable from a neutral state followed by a `:D`, and only preserved by `:D`. I expected at least a few players to intentionally bully the catgirl with negative expressions (generally `:)`, `:(`, or `>:(`, but in an upset mood, the catgirl can perceive `;)` and `:D` as inappropriate), yet even players who wanted to see the full range of expressions chose to look in the [Github repo’s assets folder](https://github.com/ksadov/ludumdare47/tree/main/src) instead of repeatedly provoking her.

All video game characters come down to some visual assets linked together by lines of code, but their behavior is complex enough that it makes sense to say “Mario takes damage from spikes” instead of “these registers are decremented and this sequence of pixels is rendered to the screen if the game is in this state”. DFA is so simple that it doesn’t even have a concept of memory: given a state and a player input the next state will always be the same, regardless of the history of inputs up to it. Thus the game relies heavily on [pareidolia](https://en.wikipedia.org/wiki/Pareidolia) to keep the player interested. I doubt that anyone would have bothered trying most combinations of inputs if the character sprite was replaced with intricately rendered landscapes or abstract shapes, but once you make it look like the player is interacting with an entity with a face, the player themselves will supply the narrative necessary to give the game depth. This cuts the developer’s workload down tremendously, which is great because I only had 48 hours to make this thing.

Most people encounter the theory of DFAs in introductory computer science courses, but the concept is derived from [McCulloch and Pitt’s simplified model of neural activity](https://www.cs.cmu.edu/~./epxing/Class/10715/reading/McCulloch.and.Pitts.pdf). I didn’t know this fact when I made DFA, but it’s neat that my highly abstracted, simplified model of human behavior ended up implemented on a highly abstracted, simplified human brain.
Title

DFA randomly displays one of three different title screens. I borrowed this trick from [LSD Dream Simulator’s cutscene animations](https://www.youtube.com/watch?v=cKBeAJphmIw). LSD signifies, apart from the obvious, “In Logic the Symbolic Dream”, “In Lust the Sensuous Dream”, “In Leisure the Sonorous Dream”, etc. No one picked up on the DFA-implementation based pun without me prompting them, but that’s ok. At least I was entertained.
Graphics

I drew the title screen, sprites and buttons in the Procreate app on an iPad Pro with an Apple Pencil. The main character’s facial proportions are indebted to [Ilya Kushinov](https://www.artstation.com/kuvshinov_ilya), who perfected the synthesis of manga babes with classical Western portraiture. The rendering was supposed to be inspired by [Moebius](https://www.juxtapoz.com/news/black-and-white-drawings-by-moebius/), but timing constraints demanded at most minimal hatching on the clothes.

Manga + Moebius is a classic combination. Moebius inspired and was inspired by multiple Japanese manga artists and animators, among them [Hayao Miyazaki](http://www.nausicaa.net/miyazaki/interviews/miyazaki_moebious.html).

![Moebius’s illustration of Miyazaki’s Nausicaa.](/images/moebiusnausicaa.jpg)

Other points of contact between Moebius and Japanese cartoon media: the work of [Taiyo Matsumoto](https://scalar.usc.edu/works/wcysft/matsumoto-taiyo-and-influences) and the anime [Dragon’s Heaven](http://www.theanimereview.com/reviews/dragonheaven.html). DFA is heir to a distinguished tradition!
Code


I wrote the [code](https://github.com/ksadov/ludumdare47) after finishing the visual assets, because I figured that it would be the less time-consuming part. Implementing a DFA is a trivial exercise, even in a jank language like JavaScript. I would have liked to implement states and transitions using [enums](https://en.wikipedia.org/wiki/Enumerated_type), but JavaScript does not support those, so I kludged together a substitute based on [this StackExchange answer](https://stackoverflow.com/questions/287903/what-is-the-preferred-syntax-for-defining-enums-in-javascript). The state transition function is a big ugly sequence of “if” blocks. I could have implemented the transition function by looking up entries in a 2d matrix indexed by states and inputs, but hindsight is 20/20.

The actual time-consuming part was making the UI look nice. Browser games are cool because they can nominally be played on any internet-enabled device, but this requires actually scaling the game to fit diverse screen sizes. Instead of coming up with different CSS for different screen layouts, I shot for a universal solution. The catgirl’s sprite scales with the size of the viewing window and the menu is pinned to the bottom, so the game can be played with any window dimensions. Buttons are fixed `4em`, which I think means 4x the the font size of the element. This is a nonsensical measure in the context of an interface with no visible font, and I wish I’d defined it relative to viewport size instead, but now I like the way it looks too much to change it. If I knew that I’d have to explain this decision in the form of a blog, I would have been more principled.