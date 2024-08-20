document.addEventListener('DOMContentLoaded', () => {
    let page = 1;
    const gallery = document.getElementById('gallery');
    const loader = document.getElementById('loader');
    const modal = document.getElementById('imageModal');
    const modalImage = document.getElementById('modalImage');
    const closeModal = document.getElementById('closeModal');

    function loadImages() {
        fetch(`/load-images/${page}`)
            .then(response => response.json())
            .then(images => {
                images.forEach(image => {
                    const imgElement = document.createElement('img');
                    imgElement.src = `/static/images/${image}`;
                    imgElement.className = "w-80 h-60 object-cover rounded-lg shadow-lg hover:scale-105 transform transition duration-300 ease-in-out cursor-pointer";
                    imgElement.addEventListener('click', () => {
                        modalImage.src = imgElement.src;
                        modal.classList.remove('hidden');
                    });
                    gallery.appendChild(imgElement);
                });
                if (images.length < 9) {
                    loader.style.display = 'none';
                }
            });
        page++;
    }

    // Load initial images
    loadImages();

    // Infinite scrolling
    window.addEventListener('scroll', () => {
        if (window.innerHeight + window.scrollY >= document.body.offsetHeight - 100) {
            loadImages();
        }
    });

    // Close modal
    closeModal.addEventListener('click', () => {
        modal.classList.add('hidden');
    });

    // Close modal when clicking outside the image
    modal.addEventListener('click', (event) => {
        if (event.target === modal) {
            modal.classList.add('hidden');
        }
    });
});
