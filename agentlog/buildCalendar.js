
const fs = require("fs");


const dateToWeekD = date => (Math.floor(date.getTime() / 86400e+3) - 4) % 7;	// monday is 0
const dateToWeeks = date => Math.floor((Math.floor(date.getTime() / 86400e+3) - 4) / 7);


const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];


const README_PATH = "./agentlog/README.md";


const abstractTitle = line => line.replace(/[\r]/g, "").replace(/^#/, ".").replace(/#/g, "    ")
	.replace(/\]\(.*\)/, "]").replace(/^[>]/, "&#xd;>").replace(/^(.{60}).+/, "$1...");


const main = () => {
	const t0 = new Date("2025/10/21");
	const now = new Date();
	const t1 = new Date(`${now.getFullYear()}/12/31`);

	const w0 = dateToWeeks(t0);
	const w1 = dateToWeeks(t1);

	let lastYear = null;
	let lastMonth = null;

	const lines = [
		"",
		"|    | 一 | 二 | 三 | 四 | 五 | 六 | 日 |",
		"| -: | -: | -: | -: | -: | -: | -: | -: |",
	];

	for (let w = w0; w <= w1; ++w) {
		const wt = new Date((w * 7 + 6 + 4) * 86400e+3);
		const year = wt.getFullYear();
		const month = wt.getMonth();

		let ym = "&ensp;";

		if (year !== lastYear)
			ym = `**${year}**&ensp;`;
		else if (month !== lastMonth)
			ym = MONTHS[month];

		lastYear = year;
		lastMonth = month;

		const fields = [ym];

		for (let d = 0; d < 7; ++d) {
			const t = new Date((w * 7 + d + 4) * 86400e+3);
			const day = t.getDate();
			const year = t.getFullYear();
			const mmdd = String(t.getMonth() + 1).padStart(2, '0') + String(t.getDate()).padStart(2, '0');

			const fileLink = `agentlog/${year}/${mmdd}.md`;

			let abstract = "";
			let milestoneLine = null;
			const past = now - t + 28800e+3; // timezone +8
			const today = past >= 0 && past < 86400e+3;

			if (fs.existsSync(fileLink)) {
				const doc = fs.readFileSync(fileLink, "utf-8");
				const ts = doc.split("\n").filter(l => l.startsWith("##") || l.startsWith("> ")).map(abstractTitle);
				abstract = ts.join("&#xd;").replace(/"/g, '\\"');

				milestoneLine = doc.match(/.*🎉 MILESTONE\b.*/);
				milestoneLine = milestoneLine ? abstractTitle(milestoneLine[0]).replace(/^.\s+/, "") : null;
			}

			// Determine the display string for the day
			let dayStr;
			if (today) {
				// Today is marked with 🌅
				dayStr = String.fromCodePoint(0x1f305);
			} else if (milestoneLine) {
				// Days with diary entries (milestones) are marked with 🚩
				abstract = milestoneLine + "&#xd;&#xd;" + abstract;
				dayStr = "🚩";
			} else if (day == 1) {
				// First day of month uses circled 1
				dayStr = "\u2776";
			} else {
				// Regular days show the date number
				dayStr = day;
			}

			if (today)
				abstract = "TODAY&#xd;&#xd;" + abstract;
			const relativeFileLink = fileLink.replace(/^agentlog\//, "");
			const field = abstract ? `[${dayStr}](${relativeFileLink} "${abstract}")` : `${dayStr}`;

			fields.push(field);
		}

		lines.push(fields.join(" | "));
	}

	const calendarContent = lines.join("\n") + "\n";

	// Read existing README.md to preserve content before ## CALENDAR
	let readme = "";
	if (fs.existsSync(README_PATH))
		readme = fs.readFileSync(README_PATH, "utf-8");

	// Find ## CALENDAR heading
	const calendarHeading = "## DEVELOPMENT CALENDAR";
	const calendarIndex = readme.indexOf(calendarHeading);

	let finalDocument;
	if (calendarIndex !== -1) {
		// Found ## CALENDAR heading, preserve content before it and append calendar
		const beforeCalendar = readme.substring(0, calendarIndex + calendarHeading.length);
		finalDocument = beforeCalendar + "\n\n" + calendarContent;
	} else {
		// No ## CALENDAR heading found, append it with calendar content
		finalDocument = readme + (readme.endsWith("\n") ? "" : "\n") + "\n" + calendarHeading + "\n\n" + calendarContent;
	}

	fs.writeFileSync(README_PATH, finalDocument);
};


main();
