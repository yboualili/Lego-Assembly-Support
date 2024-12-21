const currentUrl = window.location.href;
maeSufficient = document.getElementById('mae-sufficient');
currentStage = document.getElementById('current-stage');
detectingWrapper = document.getElementById('detecting-wrapper');
lastStage = -1

/* UI */ 

const numOfStages = 20
const pointHeight = pointWidth = window.innerHeight / numOfStages / 1.75;
stageIndexes = document.getElementsByClassName('stage-indexes')[0];

//var audio = new Audio(currentUrl + 'static/sounds/stage_completed.wav');

for (var i = 0; i < numOfStages; i++) {
    stageIndex = document.createElement('div');
    stageIndex.style.height = pointHeight + 'px';
    stageIndex.style.width = pointWidth + 'px';
    stageIndex.classList.add('stage-index')
    stageIndex.classList.add('undone')
    stageIndexes.appendChild(stageIndex);
}

function updateStage(stage) {
    console.log(stage)
    if (stage <= numOfStages) {
        updateInstructionImage(stage);
        if (stage > -1) {
            triggerStageCompleted(stage);
            setTimeout(function(){
                zoomInstructionImage();
            }, 400);
        }
        setTimeout(function(){
            updateStageIndex(stage);
        }, 350);
        //audio.play();
    }
}

function updateInstructionImage(stage) {
    instructionImage = document.getElementById('instruction-image');
    console.log(instructionImage)
    imageUrl = currentUrl + 'static/instruction_images/' + (parseInt(stage)+1) + '.png'
    instructionImage.src = imageUrl;
}

function updateStageIndex(stage) {

    // Get stage points
    stages = stageIndexes.childNodes;
    
    // Add 'done' class to all done stages
    for (var i = 0; i <= stage; i++) {
        stages[i].classList.remove('undone');
        stages[i].classList.add('done');
    }

    // Set next stage to 'in progress'
    if (stage < numOfStages) {
        stages[parseInt(stage)+1].classList.remove('undone');
        stages[parseInt(stage)+1].classList.add('in-progress');
        stages[parseInt(stage)+1].style.height = pointHeight - 6;
        stages[parseInt(stage)+1].style.width = pointHeight - 6;
    }
}

function triggerStageCompleted(stage) {
    stageCompleted = document.getElementsByClassName('stage-completed-wrapper')[0];
    stageText = document.getElementsByClassName('stage-completed')[0];
    stageText.innerHTML = "Stage X completed!".replace('X', stage);
    stageCompleted.classList.remove('blend-in');
    setTimeout(function(){
        stageCompleted.classList.add('blend-out');
        setTimeout(function(){
            stageCompleted.classList.add('blend-in');
            stageCompleted.classList.remove('blend-out');
        }, 500);
    }, 1000);
}

function zoomInstructionImage() {
    instructionImage = document.getElementsByClassName('instruction-wrapper')[0];
    instructionImage.classList.add('zoomed');
    setTimeout(function(){
        instructionImage.classList.remove('zoomed');
    }, 1500)
}

updateStage(-1);

/* STREAM HANDLING */ 

// MAE STREAM
var maeStream = new EventSource('/mae_stream');
maeStream.onmessage = function(e) {
    if (e.data == 0) {
        detectingWrapper.classList.add('blend-out');
    } else if (e.data == 1) {
        detectingWrapper.classList.remove('blend-out');
    }
}

// CURRENT STAGE STREAM
var currentStageStream = new EventSource('/current_stage_stream');
currentStageStream.onmessage = function(e) {
    currentStage = parseInt(e.data);
    console.log(lastStage, currentStage)
    if (currentStage == (lastStage + 1)) {
        lastStage = currentStage;
        updateStage(currentStage);
    }
}
