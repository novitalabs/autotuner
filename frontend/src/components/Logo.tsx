interface LogoProps {
	className?: string;
}

export function Logo({ className = "" }: LogoProps) {
	return (
		<svg
			viewBox="0 0 32 32"
			fill="none"
			xmlns="http://www.w3.org/2000/svg"
			className={className}
		>
			{/* Main upward triangle - representing optimization & improvement */}
			{/* Inspired by Novita's geometric style, forming an "A" for Autotuner */}
			<path
				d="M 16 3 L 29 27 L 19 27 L 16 21 L 13 27 L 3 27 Z"
				fill="#3ECF8E"
			/>

			{/* Tuning indicator - three horizontal bars suggesting adjustment levels */}
			{/* These represent the iterative tuning process */}
			<rect x="11" y="14" width="10" height="2" rx="0.5" fill="white" opacity="0.9" />
			<rect x="12.5" y="10" width="7" height="1.5" rx="0.5" fill="white" opacity="0.7" />
			<rect x="14" y="7" width="4" height="1.5" rx="0.5" fill="white" opacity="0.5" />
		</svg>
	);
}
