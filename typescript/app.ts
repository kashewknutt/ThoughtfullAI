import LlamaAI from 'llamaai';

const apiToken = 'LA-026c73ebc3f540b7a46809ad25721694b256d2fc7a1a482faf6896e712216cd4'; // Replace with your actual LLaMA API token
const llamaAPI = new LlamaAI(apiToken);

const WORDS_THRESHOLD = 30;
const QUESTION_DISPLAY_TIME = 10000; // 10 seconds

let text = '';
let questions: string[] = [];
let isListening = false;
let debugInfo = '';

// Toggle listening state
const toggleListening = () => {
  isListening = !isListening;
  debugInfo = `Listening toggled: ${isListening}`;
  console.log(debugInfo);
  isListening ? startRecording() : stopRecording();
};

// Update the UI in the console
const updateUI = () => {
  console.clear();
  console.log(`Thought Provoker AI`);
  console.log(`1. Toggle Listening (Current state: ${isListening ? 'Listening' : 'Not Listening'})`);
  console.log(`2. Current Text: ${text}`);
  console.log(`3. Questions: ${questions.join(', ')}`);
  console.log(`4. Debug Info: ${debugInfo}`);
};

// Handle speech recognition results
const handleSpeechResult = async (result: string) => {
  text = (text + ' ' + result).trim();
  debugInfo = `Current text: ${text}`;
  const wordCount = text.split(' ').length;

  if (wordCount >= WORDS_THRESHOLD) {
    await generateQuestion(text);
    text = ''; // Clear text after generating question
  }
  updateUI();
};

// Generate a question using the LLaMA API
const generateQuestion = async (context: string): Promise<void> => {
    debugInfo = `Generating question for context: ${context}`;
    updateUI();
  
    try {
      const question: string = await QuestionGenerator.generate(context); // Specify type as string
      questions.push(question);
      debugInfo = `Question generated: ${question}`;
      
      setTimeout(() => {
        questions.shift();
        updateUI();
      }, QUESTION_DISPLAY_TIME);
    } catch (error) {
      debugInfo = `Error generating question: ${error}`;
    }
};

// Start audio recording
// Start audio recording
const startRecording = async () => {
    console.log("Attempting to start recording...");
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      const audioChunks: Blob[] = []; // Explicitly type the array
  
      mediaRecorder.ondataavailable = (event: BlobEvent) => { // Ensure event is typed correctly
        if (event.data.size > 0) {
          audioChunks.push(event.data);
          console.log('Audio chunk captured:', event.data);
        }
      };
  
      mediaRecorder.onstart = () => {
        console.log("Recording started.");
      };
  
      mediaRecorder.onstop = async () => {
        console.log("Stopping recording...");
        if (audioChunks.length > 0) {
          console.log("Processing audio chunks...");
          await processAudioChunks(audioChunks);
        } else {
          console.warn("No audio chunks to process.");
        }
      };
  
      mediaRecorder.start();
  
      // Stop recording after 5 seconds
      setTimeout(() => {
        console.log("Stopping recording...");
        mediaRecorder.stop();
      }, 5000); // Adjust the duration as needed
  
    } catch (error) {
      console.error("Error accessing microphone:", error);
    }
  };
  
  // Process audio chunks
  const processAudioChunks = async (audioChunks: Blob[]) => {
    console.log("Processing audio chunks...");
    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    const audioFile = new File([audioBlob], 'recording.wav', { type: 'audio/wav' });
  
    console.log("Sending audio file to API...");
    try {
      const result = await query(audioFile);
      console.log("API response:", result);
      handleSpeechResult(result.text);
    } catch (error) {
      console.error('Error querying the Hugging Face API:', error);
    }
  };
  

const stopRecording = () => {
  console.log("Stopped listening.");
};

// Query the Hugging Face API for speech recognition
const query = async (audioFile: File) => {
  const response = await fetch(
    'https://api-inference.huggingface.co/models/openai/whisper-large-v3',
    {
      headers: {
        Authorization: 'Bearer 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', // Replace with your actual Hugging Face API token
      },
      method: 'POST',
      body: audioFile,
    }
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Error: ${errorText}`);
  }

  const result = await response.json();
  return result;
};

// QuestionGenerator for LLaMA API
const QuestionGenerator = {
  async generate(context: string) {
    console.log('Generating question for context:', context);
    try {
      const apiRequestJson = {
        messages: [
          { role: 'user', content: context },
        ],
        functions: [
          {
            name: "generate_question",
            description: "Generate a creative question based on the provided context",
            parameters: {
              type: "object",
              properties: {
                context: {
                  type: "string",
                  description: "The context for generating a question.",
                },
              },
              required: ["context"],
            },
          },
        ],
        stream: false,
        function_call: "generate_question",
      };

      const response = await llamaAPI.run(apiRequestJson);
      const question = response.choices[0].message.content.trim() || 'Could not generate a question.';
      console.log('Generated question:', question);
      return question;
    } catch (error) {
      console.error('Error generating question:', error);
      return 'Error generating question.';
    }
  }
};

// Start the application
updateUI();
toggleListening(); // Start listening

// Keep the event loop alive
setInterval(() => {}, 1000);

