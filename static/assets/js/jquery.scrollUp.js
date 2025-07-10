let scrollPercentage = () => {
    let scrollProgress = document.getElementById("progress");
    let progressValue = document.getElementById("progress-value");
    let pos = document.documentElement.scrollTop;
    let calcHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
    let scrollValue = Math.round(pos * 100 / calcHeight);

    scrollProgress.style.background = `conic-gradient(#01ADEF ${scrollValue}%, #c0c0ff ${scrollValue}%)`;
    progressValue.textContent = `${scrollValue}%`;

    // Show/hide progress bar based on scroll position
    if (pos > 20) {
        scrollProgress.classList.remove("hide");
        scrollProgress.classList.add("show");
    } else {
        scrollProgress.classList.remove("show");
        scrollProgress.classList.add("hide");
    }
    
    // Add event listener to scroll to top when progress bar is clicked
    scrollProgress.addEventListener("click", () => {
        window.scrollTo({ top: 0, behavior: "smooth" });
    });
};

window.onscroll = scrollPercentage;
window.onload = scrollPercentage;