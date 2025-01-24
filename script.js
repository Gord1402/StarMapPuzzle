document.addEventListener("DOMContentLoaded", async function () {
    const names = await (await fetch('names.json')).json();
 
    const imageContainer = document.getElementById('image-container');
    let r_height = Math.min(window.innerHeight, window.innerWidth) - 10;
    let imageData = []; // Store the loaded JSON data globally

    function createImage(src, x, y, width, height) {
        const img = document.createElement('img');
        img.src = src;
        img.style.position = 'absolute'; // Ensure images are positioned absolutely
        img.style.left = `${x}px`;
        img.style.top = `${y}px`;
        img.style.width = `${width}px`;
        img.style.height = `${height}px`;
        img.classList.add('draggable-image');
        return img;
    }

    function createTextElement(text, x, y) {
        const textElement = document.createElement('div');
        textElement.textContent = text;
        textElement.style.position = 'absolute';
        textElement.style.left = `${x}px`;
        textElement.style.top = `${y}px`;
        textElement.style.color = 'white';
        textElement.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
        textElement.style.padding = '5px';
        textElement.style.borderRadius = '5px';
        textElement.style.fontSize = '14px';
        textElement.style.zIndex = '1000';
        return textElement;
    }

    function spreadImages() {
        const mapWidth = r_height;
        const mapHeight = r_height;

        const freeLeft = window.innerWidth - mapWidth - 10; // 10px margin
        const freeBottom = window.innerHeight - mapHeight - 10; // 10px margin

        let spreadAreaWidth = freeLeft > 0 ? freeLeft : 0;
        let spreadAreaHeight = freeBottom > 0 ? freeBottom : 0;

        if (spreadAreaWidth >= r_height || spreadAreaHeight >= r_height) {
            // Spread images on the left or bottom
            imageData.forEach(data => {
                const img = createImage(`images/${data.name}`, 0, 0, data.w * r_height, data.h * r_height);

                let start_x, start_y;
                if (spreadAreaWidth >= r_height) {
                    start_x = (Math.random() * 0.7) + 0.1;
                    start_y = (Math.random() * 0.7) + 0.1;
                    img.style.left = `${mapWidth + start_x * spreadAreaWidth}px`;
                    img.style.top = `${start_y * mapHeight}px`;
                } else if (spreadAreaHeight >= r_height) {
                    start_x = (Math.random() * 0.7) + 0.1;
                    start_y = (Math.random() * 0.7) + 0.1;
                    img.style.left = `${start_x * mapWidth}px`;
                    img.style.top = `${mapHeight + start_y * spreadAreaHeight}px`;
                }

                setupDragAndDrop(img, data);
                imageContainer.appendChild(img);
            });
        } else {
            // Not enough space, reduce r_height by half and try again
            r_height /= 1.2;
            spreadImages();
        }
    }

    function setupDragAndDrop(img, data) {
        let isDragging = false;
        let offsetX, offsetY;

        img.addEventListener('mousedown', (e) => {
            e.preventDefault();
            isDragging = true;
            offsetX = e.clientX - img.getBoundingClientRect().left;
            offsetY = e.clientY - img.getBoundingClientRect().top;
            img.style.cursor = 'grabbing';
            img.style.zIndex = '1';
        });

        document.addEventListener('mousemove', (e) => {
            if (isDragging) {
                e.preventDefault();
                img.style.left = `${e.clientX - offsetX}px`;
                img.style.top = `${e.clientY - offsetY}px`;
            }
        });

        document.addEventListener('mouseup', (e) => {
            if (isDragging) {
                e.preventDefault();
                isDragging = false;
                img.style.zIndex = '0';
                if (Math.sqrt(Math.pow((e.clientX - offsetX) - data.x * r_height, 2) + Math.pow((e.clientY - offsetY) - data.y * r_height, 2)) < 15) {
                    img.style.left = `${data.x * r_height}px`;
                    img.style.top = `${data.y * r_height}px`;
                    img.style.zIndex = '-1';

                    // Add the glow effect
                    img.classList.add('glow-green');

                    // Remove the glow effect after 1 second
                    setTimeout(() => {
                        img.classList.remove('glow-green');
                    }, 1000);

                    // Display the data.id text
                    const textElement = createTextElement(names.filter((value) => {return value.abbr == data.id;})[0].name, data.x * r_height, data.y * r_height);
                    imageContainer.appendChild(textElement);
    
                    setTimeout(() => {
                        textElement.remove()
                    }, 3000);
                }
                img.style.cursor = 'grab';
            }
        });
    }

    // Function to move all images to their correct spots
    function moveAllImagesToCorrectSpots() {
        const images = document.querySelectorAll('.draggable-image');
        images.forEach((img, index) => {
            if (index === 0) return; // Skip the map image
            const data = imageData[index - 1];
            if (data) {
                img.style.left = `${data.x * r_height}px`;
                img.style.top = `${data.y * r_height}px`;
                img.classList.add('glow-green');
                setTimeout(() => {
                    img.classList.remove('glow-green');
                }, 1000);
                // Display the data.id text
                const textElement = createTextElement(names.filter((value) => {return value.abbr == data.id;})[0].name, data.x * r_height, data.y * r_height);
                imageContainer.appendChild(textElement);

                setTimeout(() => {
                    textElement.remove()
                }, 3000);

            }
        });
    }

    // Add event listener for the 's' key
    document.addEventListener('keydown', (e) => {
        if (e.key === 's') {
            moveAllImagesToCorrectSpots();
        }
    });

    // Load JSON data from the external file
    fetch('images/labels.json')
        .then(response => response.json())
        .then(data => {
            imageData = data; // Store the loaded JSON data
            spreadImages();
            const mapImg = createImage('images/map.png', 0, 0, r_height, r_height);
            mapImg.style.zIndex = '-2';
            imageContainer.insertBefore(mapImg, imageContainer.firstChild);
        })
        .catch(error => {
            console.error('Error loading JSON data:', error);
        });
});