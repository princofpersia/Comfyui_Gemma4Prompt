updated so now if you  *put something in *asterisk**  it will always be injected into the prompt exactly how you put it (good for lora activations or key phrases that work) 

MODE > "PREVIEW" - generates the prompt, keep pressing run on comfyui to roll prompts if your not happy,  MODE "SEND" - kill llama and the model. - now you can use the prompt and it wont take any system resources - swap back to prewview to get a new prompt. 



<img width="324" height="546" alt="image" src="https://github.com/user-attachments/assets/63d3fc14-d58d-421d-aea8-419f9ab057c9" />


issue with seed randomiser, add this 
<img width="558" height="188" alt="image" src="https://github.com/user-attachments/assets/9de34ae0-761d-4744-8e2e-00c0ec0dea10" />

## Added Gemma4 Batch Captioner custom node
For Batch captioning multiple images using gemma4 (usefull for captioning Lora training dataset)

<img width="418" height="457" alt="batch captioner" src="https://github.com/user-attachments/assets/8fe4beef-57e9-49a6-b3ff-0040415d60e7" />

### Added word count input
User can input the word count to keep control over number of words of the generated prompts.

### added custom system prompt preset + custom system prompt input
when new preset of custom prompt selected, user can input the custom system prompt in the seperate input, keeping the user input prompt clutter free




