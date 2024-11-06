// Typing effect for the headline
const text = "Welcome to the Career Guidance System!";
let index = 0;
function type() {
    if (index < text.length) {
        document.getElementById("headline").innerHTML += text.charAt(index);
        index++;
        setTimeout(type, 100);
    }
}

// Call the typing effect when the page loads
window.onload = type;
