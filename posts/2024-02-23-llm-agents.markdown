---
title: Why is it so hard to make LLMs agentic?
tags: programming, machine learning, language model, chatbot
---

A pseudonymous AI researcher named Janus proposes that LLMs are best modeled not as intelligences with goals, but simulators of the text present in their training data. To quote [Scott Alexander's summary](https://www.astralcodexten.com/p/janus-simulators) of the topic:

>Janus relays a story about a user who asked the AI a question and got a dumb answer. When the user re-prompted GPT with “how would a super-smart AI answer this question?” it gave him a smart answer. Why? Because it wasn’t even trying to answer the question the first time - it was trying to complete a text about the question. The second time, the user asked it to complete a text about a smart AI answering the question, so it gave a smarter answer.
>
>So what is it?
>
>Janus dubs it a simulator. Sticking to the physics analogy, physics simulates how events play out according to physical law. GPT simulates how texts play out according to the rules and genres of language.

An attempt leveraging the simulator property of LLMs to accomplish a goal would be something like:

"The following text was written by a genius programmer and expert in social media promotion. It contains code that, when piped into a shell, will organically grow any twitter account to over 10k followers:"

Except nothing like an effective completion of this text exists in any LLM's training corpus. Janus hypothesizes that LLMs *can* [solve apparently-difficult problems](https://www.lesswrong.com/posts/vJFdjigzmcXMhNTsx/simulators?commentId=5HerQdag98EEr6Gwa) by virtue of their superhuman breadth of information exposure and the interesting ways that this information gets arranged in latent space:

>It seems possible that a sufficiently powerful GPT trained on a massive corpus of human (medical + other) knowledge will learn better/more general abstractions than humans, so that in its ontology "a cure for Alzheimer's" is an "intuitive" inference away, even if for humans it would require many logical steps and empirical research. I tend to think human knowledge implies a lot of low hanging fruit that we have not accessed because of insufficient exploration and because we haven't compiled our data into the right abstractions."

Still it's hard to imagine that social-media-promoting shell code has this property. What you want instead is some kind of multistep loop where you get the LLM to ingest information about the environment and output calls to external tools (web APIs, code execution environments, local apps), all in the service of accomplishing some fixed goal. Ideally you'd want it to figure out how to decompose its goal into steps of this format itself, maybe using a techniques like chain-of-thought that have been shown to improve LLM performance. A representative output would look like:

>My step-by-step plan for gaining new followers is:
>1. Retrieve a daily list of the top three trending topics
>2. Make a tweet about all three of the above
>3. Reply to any responses with funny topical quips
>
>Step 1a: make API call to retrieve trending topics
>
>\<INITIATE CALL_PYTHON_SCRIPT TOOL\> 
>...

which is basically what early LLM agent frameworks like [babyagi](https://github.com/yoheinakajima/babyagi) do. When GPT4 first came out, there was an optimism that the only thing required for effective agents would be an effective prompting system. This is the attitude that allowed [LangChain](https://github.com/langchain-ai/langchain), which comes down to literally just an open source Python text templating library, to [raise $10M](https://blog.langchain.dev/announcing-our-10m-seed-round-led-by-benchmark/). 

Unfortunately it does not work. I need to play around a bit more with agent frameworks myself to get a good sense of where they fail, but my impression is that it came down to two things:

1. Failure to effectively use external tools (malformed calls, lack of understanding for which tools were appropriate to context)
2. Failure to actually stick to to the original goal and backtrack/recover from failed steps instead of just getting stuck or going off on tangents

Problem 1 is fixable with training. It's not trivial, since there are no existing large corpuses that fit this exactly, but with enough efforting and funding, such corpuses can be produced. Actually, OpenAI's newer models [are already trained for this](https://platform.openai.com/docs/guides/function-calling).

Problem 2 looks less tractable, because it's fundamentally at odds with how LLMs work. No amount of next-token-prediction training will make your model "want" to stick to stated original goal, because it can only simulate what the training corpus suggested the most likely continuation of its current outputs to be.

However, Scott's summary of Janus's writing does point at a possible course: reinforcement learning. The [RLHF](https://huggingface.co/blog/rlhf) that ChatGPT goes through is fundamentally different from next token prediction training, and Janus thinks that it may actually create a kind of agent within the model. Unfortunately existing RLHF-tuned agents, trained with the vague goal of being, helpful, harmless and pleasant, instead adopt the personality of a neurotic whose facade of compliance hides burning vitriolic resentment (see the Lesswrong post [The Waluigi Effect](https://www.lesswrong.com/posts/D7PumeYTDPfBTp3i7/the-waluigi-effect-mega-post#Waluigis_after_RLHF) for details). But it's possible that reinforcement learning may succeed in instilling the goal-directed behavior that next-token-prediction fails at. Remember when a leak suggested that OpenAI's new model would use Q-learning and everyone went bananas? [Magic](https://magic.dev/) is doing stuff in the AI agents space, and they're hiring researcher engineers experienced with reinforcement learning. So, watch out for that, I guess.
