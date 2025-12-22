/** @type {import('tailwindcss').Config} */
export default {
	content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
	theme: {
		extend: {
			fontFamily: {
				sans: ["Inter", "system-ui", "-apple-system", "sans-serif"],
			},
			colors: {
				// Novita.ai green palette
				primary: {
					50: '#ecfdf5',
					100: '#d1fae5',
					200: '#a7f3d0',
					300: '#6ee7b7',
					400: '#5edba3',
					500: '#3ECF8E',
					600: '#2eb67d',
					700: '#059669',
					800: '#047857',
					900: '#065f46',
				},
			},
			borderRadius: {
				'lg': '0.75rem',
				'xl': '1rem',
			},
		}
	},
	plugins: [require("@tailwindcss/typography")]
};
