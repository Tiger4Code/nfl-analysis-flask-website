const canvas = document.getElementById("footballField");
const ctx = canvas.getContext("2d");

// Field dimensions and properties
const fieldWidth = 1200;
const fieldHeight = 600;
const endZoneWidth = 100;
const yardLineSpacing = (fieldWidth - 2 * endZoneWidth) / 10; // spacing for each section

// Scaling factor for yards to canvas pixels
const xScale = canvas.width / 120;
const yScale = canvas.height / 53.3;

// Define triangle properties
const triangleSize = 13;

let fieldMsg = "Loading Play";
let currentEvent = 'NA';
let ball_start_x = 0;
let ball_start_y = 0;
let ball_start_pos_set = false;
let ball_frame_start = 0;
let ball_y_increment = 0;
let ball_x_increment = 0;
let ball_player_id = 0;

let speedMultiplier = 1; // Speed multiplier (1x to 3x)
let paused = false; // Pause state
let reset = false;
let animationFrameId; // To keep track of the animation frame

let selectedID = 0;


// Convert degrees to radians
function degreesToRadians(degrees) {
    return degrees * (Math.PI / 180);
}
function formatText(input) {
    return input
        .split('_') // Split the string into words at underscores
        .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()) // Capitalize each word
        .join(' '); // Join the words back with spaces
}

function drawField() {
    // Draw the field background
    ctx.fillStyle = "#007A33"; // green field color
    ctx.fillRect(0, 0, fieldWidth, fieldHeight);

    // Draw the end zones
    ctx.fillStyle = "#0033A0"; // blue end zone color
    ctx.fillRect(0, 0, endZoneWidth, fieldHeight); // left end zone
    ctx.fillRect(fieldWidth - endZoneWidth, 0, endZoneWidth, fieldHeight); // right end zone

    // Yard markers in sequence 0, 10, 20, 30, 40, 50, 40, 30, 20, 10, 0
    const yardMarkers = [0, 10, 20, 30, 40, 50, 40, 30, 20, 10, 0];

    // Draw yard lines and labels
    ctx.strokeStyle = "white";
    ctx.lineWidth = 2;
    for (let i = 0; i < yardMarkers.length; i++) {
        const x = endZoneWidth + i * yardLineSpacing;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, fieldHeight);
        ctx.stroke();

        // Add yard markers, excluding the end zone markers (0s)
        if (yardMarkers[i] !== 0) {
            ctx.font = "20px Arial";
            ctx.fillStyle = "white";
            ctx.fillText(yardMarkers[i].toString()[0] + ' ' + yardMarkers[i].toString()[1], x - 13, fieldHeight / 2 + 100); // bottom label
            ctx.fillText(yardMarkers[i].toString()[0] + ' ' + yardMarkers[i].toString()[1], x - 13, fieldHeight / 2 - 85); // top label
        }
    }

    // Draw hash marks
    for (let i = 0; i < 100; i++) {

        if(i % 2 === 0 && i % 10 !== 0){
            const x = (10 + i) * xScale;
            ctx.fillRect(x, 10, 2, 4);
            ctx.fillRect(x, 52 * yScale, 2, 4);

            ctx.fillRect(x, (fieldHeight / 2) + 50, 2, 4);
            ctx.fillRect(x, (fieldHeight / 2) - 50, 2, 4);
        }
    }
    
    const drawLineOfScrimmage = () => {
        // Draw line of Scrimmage
        ctx.strokeStyle = "yellow";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo((line_of_scrimmage[0]) * xScale, 0);
        ctx.lineTo((line_of_scrimmage[0]) * xScale, fieldHeight);
        ctx.stroke();
    } 

    const drawLineOfDownYards = (message, direction) => {
        const boxWidth = 300;
        const boxHeight = 60;
        const x = canvas.width/2 - boxWidth - 10; // 10px padding from right
        const y = 40 * yScale; // Vertical position of the box
    
        // Draw transparent box
        ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.fillRect(x, y, boxWidth, boxHeight);
    
        // Add black text
        ctx.fillStyle = 'black';
        ctx.font = '16px Arial';
        ctx.textBaseline = 'middle';
        ctx.fillText(message, x + 10, y + boxHeight / 2);
    
        // Draw direction triangle
        ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    
        // Triangle dimensions
        const triangleHeight = 60;
        const triangleWidth = 30;
        const triangleY = y + boxHeight / 2; // Center of the box vertically
    
        if (direction === 'left') {
            // Draw a triangle pointing left
            ctx.beginPath();
            ctx.moveTo(x - triangleWidth, triangleY); // Tip of the triangle
            ctx.lineTo(x, triangleY - triangleHeight / 2); // Top corner
            ctx.lineTo(x, triangleY + triangleHeight / 2); // Bottom corner
            ctx.closePath();
            ctx.fill();
        } else if (direction === 'right') {
            // Draw a triangle pointing right
            ctx.beginPath();
            ctx.moveTo(x + boxWidth + triangleWidth, triangleY); // Tip of the triangle
            ctx.lineTo(x + boxWidth, triangleY - triangleHeight / 2); // Top corner
            ctx.lineTo(x + boxWidth, triangleY + triangleHeight / 2); // Bottom corner
            ctx.closePath();
            ctx.fill();
        }
    };
    

    // Draw goal posts at each end zone
    const drawGoalPost = (x, y) => {
        ctx.strokeStyle = "yellow";
        ctx.lineWidth = 4;

        // Draw the crossbar
        ctx.beginPath();
        // Left goal post, shaped like ]
        ctx.moveTo(x + 70, y - 50);
        ctx.lineTo(x + 100, y - 50); // Crossbar extends right

        // Right goal post, shaped like [
        ctx.moveTo(x + 100, y + 50);
        ctx.lineTo(x + 70, y + 50); // Crossbar extends left
        ctx.stroke();
    };

    // Draw text
    const drawText = () => {
        const boxWidth = 500;
        const boxHeight = 30;
        const x = canvas.width - boxWidth - 10; // 10px padding from right
        const y = 30; // 10px padding from top

        if(fieldMsg != ""){
            // Draw faded white background
            const gradient = ctx.createLinearGradient(x, y, x + boxWidth, y);
            gradient.addColorStop(0, 'rgba(255, 255, 255, 0.8)');
            gradient.addColorStop(1, 'rgba(255, 255, 255, 0.2)');
            ctx.fillStyle = gradient;
            ctx.fillRect(x, y, boxWidth, boxHeight);

            // Add black text
            ctx.fillStyle = 'black';
            ctx.font = '16px Arial';
            ctx.textBaseline = 'middle';
            ctx.fillText(fieldMsg, x + 10, y + boxHeight / 2);
        }
        
    };

    // Draw team name
    const drawTeamName = (name, side) => {
        const boxWidth = 30;  // Box width is the width of the vertical text
        const boxHeight = canvas.height; // Box height is the height of the canvas
        const padding = 20;    // Padding around the text
    
        // Determine the x position based on the side parameter
        const x = side === 'left' ? padding : canvas.width - boxWidth - padding;
        
        // Set text properties
        ctx.fillStyle = side === 'left' ? 'yellow' : 'red';  // Set the text color to white
        ctx.font = 'bold 20px Arial';
        ctx.textBaseline = 'middle';
    
        // Save the current context state
        ctx.save();
    
        // Move the origin to the starting point for the vertical text
        ctx.translate(x + boxWidth / 2, canvas.height / 2); // Center the text vertically and horizontally
    
        // Rotate the context by -90 degrees to make the text vertical
        ctx.rotate(-Math.PI / 2);
    
        // Draw the text, now centered vertically and horizontally
        ctx.textAlign = 'center';
        ctx.fillText(name, 0, 0);
    
        // Restore the context to its original state
        ctx.restore();
    };
    
    drawLineOfScrimmage();
    drawLineOfDownYards(downInfo[0] + ' & ' + downInfo[1], downInfo[2]);
    drawTeamName(home_team[0], 'left');
    drawTeamName(visitor_team[0], 'right');
    drawGoalPost(endZoneWidth - 180, fieldHeight / 2); // Left goal post
    drawGoalPost(fieldWidth - endZoneWidth + 10, fieldHeight / 2); // Right goal post

    // Draw sidelines
    ctx.strokeStyle = "white";
    ctx.lineWidth = 10;

    // Top sideline
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(fieldWidth, 0);
    ctx.stroke();

    // Bottom sideline
    ctx.beginPath();
    ctx.moveTo(0, fieldHeight);
    ctx.lineTo(fieldWidth, fieldHeight);
    ctx.stroke();
    drawText();
}

// Initialize drawing
drawField();


    

// Draw a triangle representing the player
function drawPlayer(x, y, orientation, color, player_color, selected = false) {
    ctx.save();
    ctx.translate(x * xScale, y * yScale);
    ctx.rotate(degreesToRadians(orientation));

    ctx.beginPath();
    ctx.moveTo(0, -triangleSize);
    ctx.lineTo(-triangleSize / 2, triangleSize / 2);
    ctx.lineTo(triangleSize / 2, triangleSize / 2);
    ctx.closePath();

    // Fill the triangle
    ctx.fillStyle = color;
    ctx.fill();

    
    // Add a thin white border if selected
    if (selected) {
        ctx.lineWidth = 3; // Thin border
        ctx.strokeStyle = "black";
        ctx.stroke();
    }else{
        ctx.lineWidth = 3; // Thin border
        ctx.strokeStyle = player_color;
        ctx.stroke();
    }

    ctx.restore();
}

function animateAndDrawBall(x, y) {
    // Draw the ball
    ctx.beginPath();
    ctx.arc(x * xScale, y * yScale, 5, 0, Math.PI * 2);
    ctx.fillStyle = "pink";
    ctx.fill();
}


// Animation loop
function animatePlayers() {
    if (paused) return; // Skip the frame if paused

    // ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawField();

    // Iterate over each player
    players.forEach(player => {
        const { playerDataList, currentPointIndex, progress, completed, position, player_id } = player;

        if(reset){
            player.completed = false;
            player.progress = 0;
            player.currentPointIndex = 0;
            console.log('reset completed')
        }

        // If player has completed their animation, draw them at the last position and skip further updates
        if (completed) {
            const finalData = playerDataList[playerDataList.length - 1];
            drawPlayer(finalData.x, finalData.y, finalData.o, player.color, player.player_color);
            return;
        }

        // Get current and next data points for the player
        const currentData = playerDataList[currentPointIndex];
        const nextData = playerDataList[currentPointIndex + 1];

        // Interpolate position and orientation
        const x = currentData.x + (nextData.x - currentData.x) * player.progress;
        const y = currentData.y + (nextData.y - currentData.y) * player.progress;
        const orientation = currentData.o + (nextData.o - currentData.o) * player.progress;

        // Draw the player at the interpolated position with player color
        drawPlayer(x, y, orientation, player.color, player.player_color, selectedID == player_id);

        if (currentData.event !== 'NA') {
            currentEvent = currentData.event;
            console.log(currentEvent);
            fieldMsg = formatText(currentEvent);
        }
        if (player.position == 'QB' && (currentData.frame_type === 'SNAP' || currentData.frame_type === 'AFTER_SNAP')) {
            if (currentEvent === 'man_in_motion' || currentEvent === 'play_action' || currentEvent === 'ball_snap') {
                animateAndDrawBall(x, y);
            }
        }

        // Handle pass_forward event
        if (currentEvent === 'pass_forward') {
            if (player.position === 'QB' && !ball_start_pos_set) {
                ball_start_pos_set = true;
                ball_start_x = x;
                ball_start_y = y;
                ball_frame_start = currentData.frame_id;
            }

            const frames_remaining = frame_count_ball_rece - ball_frame_start;
            const ball_progress = (currentData.frame_id - ball_frame_start) / frames_remaining;

            const ball_x = ball_start_x + (play_ball_target_x - ball_start_x) * ball_progress;
            const ball_y = ball_start_y + (play_ball_target_y - ball_start_y) * ball_progress;

            animateAndDrawBall(ball_x, ball_y);
        }
        if (currentEvent === 'pass_arrived') {
            if (x === play_ball_target_x && y === play_ball_target_y) {
                ball_player_id = player_id;
            }
        }
        if (currentEvent === 'pass_outcome_caught' || currentEvent === 'pass_arrived') {
            if (ball_player_id === player_id) {
                animateAndDrawBall(x, y);
            }
        }
        // Update progress towards the next data point
        player.progress += 0.02 * speedMultiplier;

        // Move to the next data point if interpolation is complete
        if (player.progress >= 1) {
            player.progress = 0;
            player.currentPointIndex++;

            // Mark player as completed if they reach the last data point
            if (player.currentPointIndex >= playerDataList.length - 1) {
                player.completed = true;
            }
        }
    });
    if(reset){
        reset = false;
    }
    // Call the next frame
    animationFrameId = requestAnimationFrame(animatePlayers);
}



// Functions to control the animation

function pauseResumeAnimation() {
    if (paused) {
        paused = false;
        animatePlayers(); // Resume the animation loop
        document.getElementById("pauseButton").textContent = 'Pause'
    }else{
        paused = true;
        document.getElementById("pauseButton").textContent = 'Resume';
    }
}

function resetResumeAnimation(){
    reset = true;
    fieldMsg = "Loading Play";
   
}

function setSpeed(multiplier) {
    speedMultiplier = Math.min(Math.max(multiplier, 1), 3); // Clamp value between 1 and 3
}



// Start the animation
animatePlayers();



document.getElementById("pauseButton").addEventListener("click", pauseResumeAnimation);
document.getElementById("resetButton").addEventListener("click", resetResumeAnimation);
document.getElementById("speedControl").addEventListener("change", event => {
    setSpeed(parseFloat(event.target.value));
});

$(document).ready(function(){
    $('.player-select').click(function(e){
        e.preventDefault();
        selectedID = $(this).attr('data-id');
    });
    $(window).scroll(function () {
        if ($(this).scrollTop() > 400) {
            $('.fixed-canvas').addClass('fix-hold');
        } else {
            $('.fixed-canvas').removeClass('fix-hold');
        }
    });
})