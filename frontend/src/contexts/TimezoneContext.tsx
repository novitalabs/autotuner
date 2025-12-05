import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import apiClient from '../services/api';

interface TimezoneContextType {
	timezone: string;
	timezoneOffsetMs: number;
	formatDate: (date: Date | string) => string;
	formatTime: (date: Date | string) => string;
	formatDateTime: (date: Date | string) => string;
}

const TimezoneContext = createContext<TimezoneContextType>({
	timezone: 'UTC',
	timezoneOffsetMs: 0,
	formatDate: (date) => new Date(date).toLocaleDateString(),
	formatTime: (date) => new Date(date).toLocaleTimeString(),
	formatDateTime: (date) => new Date(date).toLocaleString(),
});

export const useTimezone = () => useContext(TimezoneContext);

interface TimezoneProviderProps {
	children: ReactNode;
}

export function TimezoneProvider({ children }: TimezoneProviderProps) {
	const [timezone, setTimezone] = useState<string>('UTC');
	const [timezoneOffsetMs, setTimezoneOffsetMs] = useState<number>(0);

	useEffect(() => {
		// Fetch timezone from backend
		apiClient.getSystemInfo()
			.then((info) => {
				const tz = info.timezone || 'UTC';
				setTimezone(tz);

				// Calculate timezone offset in milliseconds
				// Get the offset by comparing formatted date components
				const now = new Date();

				// Format in UTC and target timezone to get components
				const utcFormatter = new Intl.DateTimeFormat('en-US', {
					timeZone: 'UTC',
					year: 'numeric',
					month: '2-digit',
					day: '2-digit',
					hour: '2-digit',
					minute: '2-digit',
					hour12: false
				});

				const tzFormatter = new Intl.DateTimeFormat('en-US', {
					timeZone: tz,
					year: 'numeric',
					month: '2-digit',
					day: '2-digit',
					hour: '2-digit',
					minute: '2-digit',
					hour12: false
				});

				const utcStr = utcFormatter.format(now);
				const tzStr = tzFormatter.format(now);

				// Parse both as if they were in the same timezone to get the raw difference
				const utcParsed = new Date(utcStr).getTime();
				const tzParsed = new Date(tzStr).getTime();

				// The offset is the difference
				const offsetMs = tzParsed - utcParsed;
				setTimezoneOffsetMs(offsetMs);

				/*console.log('Timezone offset:', {
					timezone: tz,
					offsetMs,
					offsetHours: offsetMs / 3600000,
					utcStr,
					tzStr
				});*/
			})
			.catch((err) => {
				console.error('Failed to fetch timezone:', err);
			});
	}, []);

	const formatDate = (date: Date | string) => {
		const d = typeof date === 'string' ? new Date(date) : date;
		return d.toLocaleDateString('en-US', {
			timeZone: timezone,
			year: 'numeric',
			month: '2-digit',
			day: '2-digit'
		});
	};

	const formatTime = (date: Date | string) => {
		const d = typeof date === 'string' ? new Date(date) : date;
		return d.toLocaleTimeString('en-US', {
			timeZone: timezone,
			hour: '2-digit',
			minute: '2-digit',
			hour12: false
		});
	};

	const formatDateTime = (date: Date | string) => {
		const d = typeof date === 'string' ? new Date(date) : date;
		return d.toLocaleString('en-US', {
			timeZone: timezone,
			year: 'numeric',
			month: '2-digit',
			day: '2-digit',
			hour: '2-digit',
			minute: '2-digit',
			second: '2-digit',
			hour12: false
		});
	};

	return (
		<TimezoneContext.Provider value={{ timezone, timezoneOffsetMs, formatDate, formatTime, formatDateTime }}>
			{children}
		</TimezoneContext.Provider>
	);
}
