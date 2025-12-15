import React, { useState } from 'react';
import {
  MapPin,
  Search,
  Star,
  Clock,
  Utensils,
  Wind,
  AlertTriangle,
  Navigation,
  Heart,
  Share2,
  Phone,
  Globe,
  Check,
  LocateFixed,
  ChefHat,
  Smile,
  DollarSign,
  Sparkles,
  ToggleLeft,
  ToggleRight,
  Info,
  Map,
  X,
  Crown,
  ArrowRight
} from 'lucide-react';

// --- MOCK DATA (5 Restaurants) ---

const RESTAURANTS = [
  {
    "id": 1,
    "name": "Cafe Dulce (USC Village)",
    "matchScore": 92,
    "rating": 4.6,
    "reviews": 495,
    "distance": "N/A",
    "priceLevel": "$$",
    "isOpen": true,
    "image": "https://lh3.googleusercontent.com/gps-cs-s/AG0ilSwLzIjzWyrEYMplI2dmHKaat-X6m9kn1CSU59p0CIMPB_qPgeIO4y-4vWLbjrg1-a0vY6ig8FTdNf6mFZu9yRPXcPIzNxXzxecuq8wzSU2eg8G486YkykRlkxmlnhEPcL8hSfeo0l1LeH4i=w408-h306-k-no",
    "ai_summary": "A bustling USC Village cafe known for inventive items like the Matcha Latte and Breakfast Burrito. Customers praise the friendly staff and delicious food, often enjoying the outdoor patio despite the loud music and small space. However, be cautious of service issues, as reports mention employees being extremely rude and instances of double charging. Patrons should also expect long lines, though the queue often moves quickly.",
    "address": "3096 McClintock Ave Ste 1420, Los Angeles, CA 90007, USA",
    "phone": "(213) 536-5609",
    "website": "N/A",
    "hours": "Open Now • Closes 8:00 PM",
    "signatures": [
      { "name": "Matcha Latte", "count": 19 },
      { "name": "Breakfast Burrito", "count": 10 },
      { "name": "Pastrami Sandwitch", "count": 5 }
    ],
    "aspects": {
      "taste": {
        "rating": 4.6,
        "summary": "Inventive & Delicious Donuts/Coffee",
        "keywords": [
          { "text": "Matcha Latte", "count": 9, "type": "pos" },
          { "text": "Spicy Chicken Sandwich", "count": 2, "type": "pos" },
          { "text": "Kale Peanut Salad", "count": 2, "type": "pos" },
          { "text": "Too Sweet", "count": 3, "type": "neg" },
          { "text": "Nothing Special", "count": 2, "type": "neg" },
          { "text": "Missing Meat", "count": 1, "type": "neg" }
        ]
      },
      "service": {
        "rating": 4.3,
        "summary": "Efficient, but Inconsistent Staff Behavior",
        "keywords": [
          { "text": "Friendly Staff", "count": 6, "type": "pos" },
          { "text": "Fast Service", "count": 2, "type": "pos" },
          { "text": "Polite Service", "count": 1, "type": "pos" },
          { "text": "Extremely Rude", "count": 2, "type": "neg" },
          { "text": "Very Disrespectful", "count": 1, "type": "neg" },
          { "text": "Double Charged", "count": 1, "type": "neg" }
        ]
      },
      "environment": {
        "rating": 4.3,
        "summary": "Lively College Vibe, Often Loud",
        "keywords": [
          { "text": "Outdoor Patio", "count": 4, "type": "pos" },
          { "text": "Vibrant And Welcoming", "count": 1, "type": "pos" },
          { "text": "Lively", "count": 2, "type": "pos" },
          { "text": "Too Loud", "count": 2, "type": "neg" },
          { "text": "Small And Noisy", "count": 1, "type": "neg" },
          { "text": "Limited Seating", "count": 1, "type": "neg" }
        ]
      },
      "waiting_time": {
        "rating": 3.5,
        "summary": "Long Lines, Fast Turnover",
        "keywords": [
          { "text": "Moves Quickly", "count": 2, "type": "pos" },
          { "text": "Up To 10 Min", "count": 2, "type": "pos" },
          { "text": "Fast", "count": 1, "type": "pos" },
          { "text": "10-30 Min", "count": 3, "type": "neg" },
          { "text": "Line Is Always Long", "count": 2, "type": "neg" },
          { "text": "Wait Can Be Long", "count": 1, "type": "neg" }
        ]
      },
      "price": {
        "rating": 4.0,
        "summary": "Affordable for the Location",
        "keywords": [
          { "text": "Reasonably Priced", "count": 2, "type": "pos" },
          { "text": "Not Horrible Pricing", "count": 1, "type": "pos" },
          { "text": "Affordable", "count": 1, "type": "pos" },
          { "text": "Over Priced", "count": 1, "type": "neg" },
          { "text": "Too Expensive", "count": 1, "type": "neg" },
          { "text": "Different Prices", "count": 1, "type": "neg" }
        ]
      }
    }
  },
  {
    "id": 2,
    "name": "YGF Malatang",
    "matchScore": 95,
    "rating": 4.6,
    "reviews": 264,
    "distance": "N/A",
    "priceLevel": "$$",
    "isOpen": false,
    "image": "https://lh3.googleusercontent.com/p/AF1QipPPRCUeki0SLF2Ix7Df80tE6Io28zOz2ooFXFhW=w408-h544-k-no",
    "ai_summary": "This top-rated Malatang is highly regarded for its delicious food and classic options like Beef Bone Broth. Service is generally praised as friendly and efficient, often featuring no wait time. However, potential diners should note severe hygiene complaints, with multiple reports of diarrhea and concerns about worst food quality, alongside instances of rude staff.",
    "address": "2526 S Figueroa St, Los Angeles, CA 90007, USA",
    "phone": "(949) 345-9685",
    "website": "https://www.ygfmalatangca.com/",
    "hours": "Closed • Opens 11:00 AM (Today: 11:00 AM – 2:00 AM)",
    "signatures": [
      { "name": "Malatang / Spicy Hot Pot", "count": 59 },
      { "name": "Beef Bone Broth", "count": 5 },
      { "name": "Tomato Soup", "count": 2 }
    ],
    "aspects": {
      "taste": {
        "rating": 4.7,
        "summary": "Exceptional Flavor & Variety",
        "keywords": [
          { "text": "Delicious Food", "count": 16, "type": "pos" },
          { "text": "Beef Bone Broth", "count": 5, "type": "pos" },
          { "text": "Fresh Ingredients", "count": 3, "type": "pos" },
          { "text": "Diarrhea", "count": 4, "type": "neg" },
          { "text": "Worst Food", "count": 4, "type": "neg" },
          { "text": "Stomach Problems", "count": 3, "type": "neg" }
        ]
      },
      "service": {
        "rating": 4.4,
        "summary": "Quick Self-Service, Staff Inconsistency",
        "keywords": [
          { "text": "Excellent Service", "count": 3, "type": "pos" },
          { "text": "Friendly Staff", "count": 6, "type": "pos" },
          { "text": "Staff Are Efficient", "count": 2, "type": "pos" },
          { "text": "Rude", "count": 3, "type": "neg" },
          { "text": "Missing Items", "count": 3, "type": "neg" },
          { "text": "Double Charged", "count": 1, "type": "neg" }
        ]
      },
      "environment": {
        "rating": 4.5,
        "summary": "Clean, Modern, but Lively Noise",
        "keywords": [
          { "text": "Clean", "count": 4, "type": "pos" },
          { "text": "Spacious", "count": 3, "type": "pos" },
          { "text": "Good Location Environment", "count": 2, "type": "pos" },
          { "text": "Loud", "count": 4, "type": "neg" },
          { "text": "Small", "count": 2, "type": "neg" },
          { "text": "Dirty", "count": 1, "type": "neg" }
        ]
      },
      "waiting_time": {
        "rating": 4.0,
        "summary": "Quick Turnover/No Wait",
        "keywords": [
          { "text": "No Wait", "count": 7, "type": "pos" },
          { "text": "Moves Quickly", "count": 2, "type": "pos" },
          { "text": "Fast Service", "count": 1, "type": "pos" },
          { "text": "Wait", "count": 4, "type": "neg" },
          { "text": "Line", "count": 3, "type": "neg" },
          { "text": "Long Wait", "count": 1, "type": "neg" }
        ]
      },
      "price": {
        "rating": 4.0,
        "summary": "Good Value for Price",
        "keywords": [
          { "text": "$20–30", "count": 17, "type": "pos" },
          { "text": "Reasonable", "count": 3, "type": "pos" },
          { "text": "Fair Price", "count": 3, "type": "pos" },
          { "text": "$30–50", "count": 5, "type": "neg" },
          { "text": "Expensive", "count": 2, "type": "neg" },
          { "text": "Overpriced", "count": 1, "type": "neg" }
        ]
      }
    }
  },
  {
    "id": 3,
    "name": "Thai by Trio",
    "matchScore": 90,
    "rating": 4.3,
    "reviews": 895,
    "distance": "N/A",
    "priceLevel": "$$",
    "isOpen": false,
    "image": "https://lh3.googleusercontent.com/p/AF1QipMCxP5Iqxa2nGLxZlUlvz3msx083f9Ik7nMDc1T=w425-h240-k-no",
    "ai_summary": "A popular Thai eatery with a cozy atmosphere, highly praised for its fast service and classic dishes like Pad Thai and Tom Yum Soup. Diners generally find the price to be in the mid-range and the value worth it. Despite many positive reviews, there are reports of inconsistent food quality, with issues like being too spicy or watery. Service complaints sometimes cite the worst service ever, though staff are generally found to be friendly.",
    "address": "2700 S Figueroa St #101, Los Angeles, CA 90007, USA",
    "phone": "(213) 536-5699",
    "website": "https://thaibytrio.com/",
    "hours": "Closed • Opens 11:00 AM (Today: 11:00 AM – 10:00 PM)",
    "signatures": [
      { "name": "Pad Thai", "count": 11 },
      { "name": "Tom Yum Soup", "count": 7 },
      { "name": "Chicken Satay", "count": 3 }
    ],
    "aspects": {
      "taste": {
        "rating": 4.1,
        "summary": "Inconsistent Quality, High Potential",
        "keywords": [
          { "text": "Delicious Food", "count": 11, "type": "pos" },
          { "text": "Great Flavor", "count": 3, "type": "pos" },
          { "text": "Authentic Thai Food", "count": 2, "type": "pos" },
          { "text": "Worst Thai Food", "count": 3, "type": "neg" },
          { "text": "Too Spicy", "count": 3, "type": "neg" },
          { "text": "Watery", "count": 2, "type": "neg" }
        ]
      },
      "service": {
        "rating": 4.0,
        "summary": "Friendly but Problematic",
        "keywords": [
          { "text": "Friendly Staff", "count": 9, "type": "pos" },
          { "text": "Excellent Service", "count": 3, "type": "pos" },
          { "text": "Warm And Efficient", "count": 1, "type": "pos" },
          { "text": "Worst Service Ever", "count": 4, "type": "neg" },
          { "text": "Rude", "count": 3, "type": "neg" },
          { "text": "Refused to Give Refund", "count": 1, "type": "neg" }
        ]
      },
      "environment": {
        "rating": 4.4,
        "summary": "Cozy and Inviting",
        "keywords": [
          { "text": "Cozy", "count": 3, "type": "pos" },
          { "text": "Clean", "count": 4, "type": "pos" },
          { "text": "Nice Ambience", "count": 5, "type": "pos" },
          { "text": "Loud", "count": 2, "type": "neg" },
          { "text": "No Wifi", "count": 1, "type": "neg" },
          { "text": "Atmosphere Was Empty", "count": 1, "type": "neg" }
        ]
      },
      "waiting_time": {
        "rating": 4.5,
        "summary": "Fast Service/No Wait",
        "keywords": [
          { "text": "No Wait", "count": 4, "type": "pos" },
          { "text": "Fast Turnover", "count": 1, "type": "pos" },
          { "text": "Lightning Fast", "count": 1, "type": "pos" },
          { "text": "10-30 Min", "count": 2, "type": "neg" },
          { "text": "Wait", "count": 2, "type": "neg" },
          { "text": "Rushed Out", "count": 1, "type": "neg" }
        ]
      },
      "price": {
        "rating": 4.0,
        "summary": "Mid-Range Pricing",
        "keywords": [
          { "text": "$20–30", "count": 10, "type": "pos" },
          { "text": "Worth It", "count": 3, "type": "pos" },
          { "text": "Reasonable", "count": 2, "type": "pos" },
          { "text": "$50–100", "count": 4, "type": "neg" },
          { "text": "Expensive", "count": 2, "type": "neg" },
          { "text": "Overpriced", "count": 2, "type": "neg" }
        ]
      }
    }
  },
  {
    "id": 4,
    "name": "Yunomi Handroll",
    "matchScore": 96,
    "rating": 4.5,
    "reviews": 266,
    "distance": "N/A",
    "priceLevel": "$$$",
    "isOpen": false,
    "image": "https://lh3.googleusercontent.com/p/AF1QipN51OFk_fEotwFirflvSiZaEDU1OkrAGujSP8xk=w408-h272-k-no",
    "ai_summary": "This handroll is acclaimed for its very fresh ingredients and amazing food, including Yellowtail Serrano Sashimi and Rock Shrimp Tempura. The service is frequently rated as very fast and excellent. While the cozy atmosphere is enjoyed, customers warn of long waits, sometimes over an hour, due to limited bar seating. Negative feedback is centered on comfort issues (e.g., uncomfortable chairs and tight spacing) and a few reports of poor customer service during peak rush.",
    "address": "806 E 3rd St #100, Los Angeles, CA 90013, USA",
    "phone": "(213) 988-7076",
    "website": "http://www.yunomihandroll.com/",
    "hours": "Closed • Opens 12:00 PM (Today: 12:00 PM – 9:30 PM)",
    "signatures": [
      { "name": "5 Hand Rolls Set", "count": 22 },
      { "name": "Yellowtail Serrano Sashimi", "count": 18 },
      { "name": "Rock Shrimp Tempura", "count": 16 }
    ],
    "aspects": {
      "taste": {
        "rating": 4.6,
        "summary": "Extremely Fresh Ingredients, Outstanding Quality",
        "keywords": [
          { "text": "Very Fresh", "count": 5, "type": "pos" },
          { "text": "Amazing Handrolls", "count": 4, "type": "pos" },
          { "text": "Soft Shell Crab", "count": 3, "type": "pos" },
          { "text": "Too Much Rice", "count": 3, "type": "neg" },
          { "text": "Low Quality", "count": 2, "type": "neg" },
          { "text": "Hot Rice", "count": 1, "type": "neg" }
        ]
      },
      "service": {
        "rating": 4.6,
        "summary": "Quick and Attentive Service",
        "keywords": [
          { "text": "Very Fast", "count": 4, "type": "pos" },
          { "text": "Excellent Service", "count": 4, "type": "pos" },
          { "text": "Top Notch", "count": 2, "type": "pos" },
          { "text": "Poor Customer Service", "count": 2, "type": "neg" },
          { "text": "Rushed Out", "count": 2, "type": "neg" },
          { "text": "Low Par", "count": 1, "type": "neg" }
        ]
      },
      "environment": {
        "rating": 4.5,
        "summary": "Cozy and Modern Bar Ambiance",
        "keywords": [
          { "text": "Bar Seating", "count": 10, "type": "pos" },
          { "text": "Cozy", "count": 5, "type": "pos" },
          { "text": "Clean", "count": 5, "type": "pos" },
          { "text": "Tight Spacing", "count": 3, "type": "neg" },
          { "text": "Uncomfortable Chairs", "count": 2, "type": "neg" },
          { "text": "Stiff Stools", "count": 1, "type": "neg" }
        ]
      },
      "waiting_time": {
        "rating": 3.0,
        "summary": "Slow Seating, but Fast Food Service",
        "keywords": [
          { "text": "Short Line", "count": 1, "type": "pos" },
          { "text": "3 Minutes", "count": 1, "type": "pos" },
          { "text": "Quick", "count": 3, "type": "pos" },
          { "text": "10-30 Min", "count": 4, "type": "neg" },
          { "text": "Over An Hour", "count": 2, "type": "neg" },
          { "text": "Wait To Be Seated", "count": 2, "type": "neg" }
        ]
      },
      "price": {
        "rating": 4.0,
        "summary": "Higher Priced, but Worth the Value",
        "keywords": [
          { "text": "Worth It", "count": 5, "type": "pos" },
          { "text": "$20–30", "count": 7, "type": "pos" },
          { "text": "Not Bad", "count": 2, "type": "pos" },
          { "text": "$50–100", "count": 4, "type": "neg" },
          { "text": "Overpriced", "count": 1, "type": "neg" },
          { "text": "Charge Additional Fees", "count": 1, "type": "neg" }
        ]
      }
    }
  }
];


// --- COMPONENTS ---

// 1. Info Overlay Modal (Glassmorphism)
const InfoOverlay = ({ item, isOpen, onClose }) => {
  return (
    <div className={`overflow-hidden transition-all duration-700 ease-[cubic-bezier(0.34,1.56,0.64,1)] ${isOpen ? 'max-h-96 opacity-100 mb-6' : 'max-h-0 opacity-0 mb-0'}`}>
      <div className="bg-white/80 backdrop-blur-lg rounded-2xl p-5 border border-white/50 shadow-xl mx-1 relative">

        <div className="flex justify-between items-center mb-4 border-b border-black/5 pb-2 relative z-10">
           <h3 className="font-serif text-sm font-bold text-slate-900 flex items-center gap-2 uppercase tracking-widest">
             <Navigation size={14} className="text-blue-600" /> Essentials
           </h3>
           <button onClick={onClose} className="text-slate-500 hover:text-slate-900 transition-colors bg-white/30 p-1 rounded-full">
             <X size={16} />
           </button>
        </div>

        <div className="grid grid-cols-2 gap-4 relative z-10">
          <div className="space-y-1">
            <div className="flex items-center gap-1.5 text-slate-500 mb-0.5">
              <Clock size={12} />
              <span className="text-[10px] font-bold uppercase">Hours</span>
            </div>
            <p className={`text-sm font-bold font-serif ${item.isOpen ? 'text-emerald-700' : 'text-rose-600'}`}>{item.hours}</p>
          </div>

          <div className="space-y-1">
            <div className="flex items-center gap-1.5 text-slate-500 mb-0.5">
              <Phone size={12} />
              <span className="text-[10px] font-bold uppercase">Phone</span>
            </div>
            <p className="text-sm font-medium text-slate-800 truncate">{item.phone}</p>
          </div>

          <div className="col-span-2 space-y-1">
            <div className="flex items-center gap-1.5 text-slate-500 mb-0.5">
              <MapPin size={12} />
              <span className="text-[10px] font-bold uppercase">Address</span>
            </div>
            <p className="text-sm font-medium text-slate-700 leading-snug">{item.address}</p>
          </div>

          <div className="col-span-2 space-y-1">
             <a href={`https://${item.website}`} className="flex items-center justify-between w-full bg-white/40 hover:bg-white/60 p-2 rounded-xl transition-all border border-white/30 shadow-sm group/link">
               <div className="flex items-center gap-2 text-xs font-bold text-slate-600">
                 <Globe size={12} /> Website
               </div>
               <ArrowRight size={12} className="text-slate-400 group-hover/link:text-blue-600 group-hover/link:translate-x-0.5 transition-all" />
             </a>
          </div>
        </div>
      </div>
    </div>
  );
};

// 2. Glassy Detail Tag
const DetailTag = ({ text, type }) => {
  const isPos = type === 'pos';
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-lg text-[10px] font-bold leading-none backdrop-blur-sm transition-all hover:scale-105 border
      ${isPos
        ? 'bg-emerald-100/40 border-emerald-200/50 text-emerald-800'
        : 'bg-rose-100/40 border-rose-200/50 text-rose-800'}
    `}>
      {text}
    </span>
  );
};

// 3. Liquid Bento Card (Detailed)
const BentoAspectCard = ({ label, icon, data, isMain }) => {
  if (!data) return null;

  const isHigh = data.rating >= 4.0;
  const isLow = data.rating < 3.0;

  const ratingColor = isHigh ? "text-emerald-700" : isLow ? "text-rose-700" : "text-amber-700";

  // Liquid Gradients Background
  const gradientBg = isHigh
    ? "bg-gradient-to-br from-emerald-50/80 to-teal-50/30 border-emerald-100/50"
    : isLow
      ? "bg-gradient-to-br from-rose-50/80 to-orange-50/30 border-rose-100/50"
      : "bg-gradient-to-br from-amber-50/80 to-yellow-50/30 border-amber-100/50";

  // Filter and limit keywords
  const posKeywords = data.keywords.filter(k => k.type === 'pos').slice(0, 3);
  const negKeywords = data.keywords.filter(k => k.type === 'neg').slice(0, 3);

  return (
    <div className={`
      ${gradientBg} backdrop-blur-sm rounded-2xl p-3 border flex flex-col shadow-sm relative overflow-hidden
      ${isMain ? 'col-span-2' : 'col-span-1'}
    `}>
      {/* Gloss Shine */}
      <div className="absolute -top-10 -right-10 w-20 h-20 bg-white/30 rounded-full blur-xl pointer-events-none" />

      {/* Header */}
      <div className="flex justify-between items-start mb-2 relative z-10">
        <div className="flex items-center gap-1.5 text-slate-500/90">
          {icon} <span className="text-[10px] font-bold uppercase tracking-wider">{label}</span>
        </div>
        <div className={`text-lg font-bold font-serif leading-none ${ratingColor} drop-shadow-sm`}>
          {data.rating}
        </div>
      </div>

      {/* Content Flow: Positives then Negatives */}
      <div className="flex flex-col gap-1.5 relative z-10">

         {/* Positives */}
         {posKeywords.length > 0 && (
           <div className="flex flex-wrap gap-1">
             {posKeywords.map((kw, i) => (
               <DetailTag key={`pos-${i}`} text={kw.text} type="pos" />
             ))}
           </div>
         )}

         {/* Negatives */}
         {negKeywords.length > 0 && (
           <div className="flex flex-wrap gap-1">
             {negKeywords.map((kw, i) => (
               <DetailTag key={`neg-${i}`} text={kw.text} type="neg" />
             ))}
           </div>
         )}

         {/* Fallback */}
         {posKeywords.length === 0 && negKeywords.length === 0 && (
           <span className="text-[9px] text-slate-400 italic">Balanced experience</span>
         )}
      </div>
    </div>
  );
};

// 4. Glass Ticket
const SignatureTicket = ({ name, count, rank }) => {
  let rankColor = "text-slate-300";
  let badgeIcon = null;
  let cardStyle = "bg-white/60 border-white/60";

  if (rank === 1) {
    rankColor = "text-yellow-600";
    badgeIcon = <Crown size={10} className="fill-yellow-400 text-yellow-600" />;
    cardStyle = "bg-yellow-50/40 border-yellow-200/50";
  }
  else if (rank === 2) { rankColor = "text-slate-500"; cardStyle = "bg-slate-50/40 border-slate-200/50"; }
  else if (rank === 3) { rankColor = "text-amber-700"; cardStyle = "bg-orange-50/40 border-orange-200/50"; }

  return (
    <div className={`
      flex-shrink-0 w-36 ${cardStyle} backdrop-blur-md border rounded-2xl p-3 flex flex-col justify-between
      shadow-[0_4px_12px_-2px_rgba(0,0,0,0.05)] snap-start h-20 relative overflow-hidden group hover:shadow-md transition-all
    `}>
      <div className="flex justify-between items-start">
        <div className={`font-serif font-bold text-base leading-none flex items-center gap-1 ${rankColor}`}>
          {badgeIcon} <span>0{rank}</span>
        </div>
        <span className="text-[9px] font-bold text-slate-600 bg-white/50 px-1.5 py-0.5 rounded-full border border-white/50">
          {count} <span className="opacity-50">likes</span>
        </span>
      </div>

      <div className="text-xs font-bold text-slate-800 leading-tight line-clamp-1 mt-auto relative z-10">
        {name}
      </div>
    </div>
  );
};

// 5. Liquid Main Card
const RestaurantCard = ({ item, index }) => {
  const [showInfo, setShowInfo] = useState(false);

  return (
    <div
      className="
        bg-white/70 backdrop-blur-2xl rounded-[32px] overflow-hidden
        shadow-[0_8px_32px_rgba(0,0,0,0.08)] border border-white/60
        mb-8 mx-auto flex flex-col animate-slideUp relative
      "
      style={{ animationDelay: `${index * 100}ms` }}
    >

      {/* 1. HERO IMAGE */}
      <div className="w-full h-64 relative flex-shrink-0 group cursor-pointer">
        <img src={item.image} alt={item.name} className="w-full h-full object-cover" />
        <div className="absolute inset-0 bg-gradient-to-t from-slate-900/80 via-transparent to-transparent" />

        {/* Match Score - Floating Pill */}
        <div className="absolute top-5 left-5">
           <div className="bg-black/40 backdrop-blur-xl text-white text-xs font-bold px-3 py-1.5 rounded-full shadow-lg border border-white/10 flex items-center gap-1.5">
             <Sparkles size={12} className="text-emerald-400 fill-emerald-400" /> {item.matchScore}% Match
           </div>
        </div>

        {/* Info Button - Glass Orb on Edge (Top Right) */}
        <button
          onClick={(e) => { e.stopPropagation(); setShowInfo(!showInfo); }}
          className={`
            absolute top-5 right-5 w-10 h-10 backdrop-blur-md rounded-full flex items-center justify-center
            border-2 border-white/20 transition-all duration-300 shadow-lg
            ${showInfo
              ? 'bg-slate-800 text-white rotate-180 shadow-inner'
              : 'bg-black/30 text-white hover:bg-black/50'}
          `}
        >
          {showInfo ? <X size={18} /> : <Info size={18} />}
        </button>

        {/* Header Title */}
        <div className="absolute bottom-0 left-0 w-full p-6 text-white">
           <div className="w-full pr-12">
              <h3 className="font-serif text-3xl font-bold leading-tight mb-2 shadow-black drop-shadow-md tracking-wide">
                {item.name}
              </h3>
              <div className="flex items-center gap-3 text-xs font-medium text-slate-100/90">
                <span className="flex items-center gap-1 font-bold text-slate-900 bg-white/90 px-2 py-0.5 rounded-lg backdrop-blur-md shadow-lg">
                  <Star size={10} className="fill-yellow-400 text-yellow-400" /> {item.rating}
                </span>
                <span className="bg-black/20 px-2 py-0.5 rounded-lg backdrop-blur-sm border border-white/10">{item.priceLevel}</span>
                <span className="bg-black/20 px-2 py-0.5 rounded-lg backdrop-blur-sm border border-white/10">{item.distance}</span>
              </div>
           </div>
        </div>
      </div>

      {/* 2. CONTENT BODY */}
      <div className="flex-1 p-6 flex flex-col pt-8 relative">

        {/* Background Blur Element */}
        <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-b from-white/40 to-white/10 pointer-events-none" />

        {/* EXPANDABLE INFO (Slides down with blur) */}
        <div className="relative z-10">
          <InfoOverlay item={item} isOpen={showInfo} onClose={() => setShowInfo(false)} />
        </div>

        {/* SECTION 1: Ranked Signatures */}
        <div className="mb-8 relative z-10">
          <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-3 flex items-center gap-1.5 pl-1">
             <ChefHat size={12} /> Top Signatures
          </h4>
          <div className="flex gap-3 overflow-x-auto scrollbar-hide pb-4 -mx-6 px-6 snap-x">
             {item.signatures.map((dish, idx) => (
               <SignatureTicket key={idx} rank={idx + 1} name={dish.name} count={dish.count} />
             ))}
          </div>
        </div>

        {/* SECTION 2: Deep Analysis (Liquid Bento - Detailed) */}
        <div className="mb-8 relative z-10">
           <div className="flex justify-between items-center mb-4 px-1">
              <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Deep Analysis</h4>
              <span className="text-[9px] font-bold text-indigo-500 bg-indigo-50/50 border border-indigo-100 px-2 py-0.5 rounded-full backdrop-blur-sm">AI Generated</span>
           </div>

           <div className="grid grid-cols-2 gap-3">
             <BentoAspectCard isMain={true} label="Taste" icon={<Utensils size={14} />} data={item.aspects.taste} />
             <BentoAspectCard label="Value" icon={<DollarSign size={14} />} data={item.aspects.price} />
             <BentoAspectCard label="Service" icon={<Smile size={14} />} data={item.aspects.service} />
             <BentoAspectCard label="Wait" icon={<Clock size={14} />} data={item.aspects.waiting_time} />
             <BentoAspectCard label="Vibe" icon={<Wind size={14} />} data={item.aspects.environment} />
           </div>
        </div>

        {/* SECTION 3: AI Verdict */}
        <div className="relative z-10 mb-8">
          <div className="pl-5 border-l-2 border-emerald-400/50 py-1 relative">
            <div className="absolute -left-[5px] top-0 w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.8)]"></div>
            <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-2">Verdict</h4>
            <p className="font-serif text-sm text-slate-700 italic leading-relaxed">
              "{item.ai_summary}"
            </p>
          </div>
        </div>

        {/* SECTION 4: Liquid Button */}
        <button className="
          relative z-10 w-full py-4 rounded-2xl text-sm font-bold flex items-center justify-center gap-2 uppercase tracking-widest
          text-white bg-gradient-to-r from-slate-900 to-slate-800
          shadow-[0_10px_20px_-5px_rgba(0,0,0,0.2)]
          hover:shadow-[0_15px_25px_-5px_rgba(0,0,0,0.3)] hover:-translate-y-0.5
          transition-all active:scale-[0.98] border border-white/10
        ">
          <Map size={16} /> View on Google Map
        </button>

      </div>
    </div>
  );
};

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }
  static getDerivedStateFromError(error) { return { hasError: true }; }
  componentDidCatch(error, errorInfo) { console.error(error, errorInfo); }
  render() { if (this.state.hasError) return <h1>Something went wrong.</h1>; return this.props.children; }
}

export default function SpotLite() {
  const [view, setView] = useState('home');
  const [searchQuery, setSearchQuery] = useState('');
  const [isOpenNow, setIsOpenNow] = useState(false);

  const handleSearch = (e) => {
    if(e) e.preventDefault();
    if (!searchQuery) return;
    setView('loading');
    setTimeout(() => setView('results'), 1500);
  };

  // --- SCREENS ---

  const HomeScreen = () => (
    <div className="min-h-screen relative flex flex-col items-center justify-center overflow-hidden bg-black">

      {/* 1. Background Image: Fine Dining (Liquid Luxury) */}
      <div className="absolute inset-0 z-0">
        <img
          src="https://images.unsplash.com/photo-1559339352-11d035aa65de?auto=format&fit=crop&q=80&w=1920"
          alt="Liquid Luxury Dining"
          className="w-full h-full object-cover opacity-80"
        />
      </div>

      {/* 2. Liquid & Glass Effects Layers */}
      {/* Strong Blur to create 'seen through liquid' effect */}
      <div className="absolute inset-0 z-0 backdrop-blur-md bg-black/40" />

      {/* Noise Texture */}
      <div className="absolute inset-0 z-0 opacity-20 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] mix-blend-overlay" />

      {/* Vignette & Gradient */}
      <div className="absolute inset-0 z-0 bg-gradient-to-b from-black/60 via-transparent to-black/90" />

      <div className="relative z-10 w-full max-w-md px-6 text-center animate-fadeIn">

        <h1 className="font-serif text-6xl font-bold text-transparent bg-clip-text bg-gradient-to-b from-white via-white to-white/70 mb-4 tracking-tight drop-shadow-2xl">
          SpotLite
        </h1>
        <p className="text-white/80 mb-12 text-lg font-medium tracking-wide drop-shadow-lg">
          Dining curated by AI.
        </p>

        <form onSubmit={handleSearch} className="bg-white/10 backdrop-blur-3xl p-2 rounded-[2rem] shadow-[0_8px_32px_rgba(0,0,0,0.5)] border border-white/20 transition-all hover:bg-white/20 hover:scale-[1.01]">
          <div className="relative flex items-center border-b border-white/10 mb-2">
            <MapPin className="text-white/70 ml-4" size={20} />
            <input
              type="text"
              placeholder="Paste Google Maps Link..."
              className="w-full p-4 outline-none text-white bg-transparent placeholder:text-white/50 font-medium text-base"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            <button
              type="button"
              onClick={() => setSearchQuery('Current Location: USC Campus')}
              className="mr-2 p-2 text-white/70 hover:text-amber-400 transition-colors"
            >
              <LocateFixed size={20} />
            </button>
          </div>

          <div className="flex items-center justify-between px-4 py-2">
             <button
               type="button"
               onClick={() => setIsOpenNow(!isOpenNow)}
               className="flex items-center gap-2 group"
             >
               <div className={`transition-colors ${isOpenNow ? 'text-amber-400' : 'text-white/60'}`}>
                 {isOpenNow ? <ToggleRight size={28} /> : <ToggleLeft size={28} />}
               </div>
               <span className={`text-xs font-bold tracking-widest uppercase transition-colors ${isOpenNow ? 'text-amber-400' : 'text-white/60'}`}>
                 Open Now
               </span>
             </button>

             <button
                type="submit"
                disabled={!searchQuery}
                className={`px-8 py-3 rounded-xl font-bold tracking-widest text-xs uppercase transition-all shadow-lg border border-white/10
                  ${searchQuery ? 'bg-white text-black hover:bg-amber-50' : 'bg-black/40 text-white/30 cursor-not-allowed'}
                `}
              >
                Analyze
              </button>
          </div>
        </form>
      </div>
    </div>
  );

  const LoadingScreen = () => (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-slate-50 to-stone-100">
      <div className="w-16 h-16 border-4 border-white/50 border-t-emerald-500 rounded-full animate-spin mb-6 shadow-lg" />
      <h2 className="font-serif text-xl text-slate-900 tracking-wide font-bold">Analyzing Palate...</h2>
    </div>
  );

  const ResultsView = () => (
    <div className="min-h-screen pb-20 relative bg-slate-50">
      {/* Liquid Background for Results */}
      <div className="fixed inset-0 bg-gradient-to-br from-stone-100 via-white to-blue-50 pointer-events-none" />
      <div className="fixed inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-10 pointer-events-none"></div>

      <div className="sticky top-0 z-30 px-4 py-4 flex justify-between items-center mb-4 backdrop-blur-xl bg-white/70 border-b border-white/50 shadow-sm">
        <h2 className="font-serif text-xl font-bold text-slate-900">Results</h2>
        <button onClick={() => setView('home')} className="text-[10px] font-bold text-slate-500 uppercase tracking-wider hover:text-slate-900">New Search</button>
      </div>

      <div className="px-4 max-w-xl mx-auto relative z-10">
        <p className="text-xs text-slate-400 uppercase tracking-widest mb-8 text-center flex items-center justify-center gap-2">
          <Sparkles size={12} className="text-amber-500" /> Found {RESTAURANTS.filter(r => !isOpenNow || r.isOpen).length} matches
        </p>

        {RESTAURANTS.filter(r => !isOpenNow || r.isOpen).map((item, idx) => (
          <RestaurantCard key={item.id} item={item} index={idx} />
        ))}
      </div>
    </div>
  );

  return (
    <ErrorBoundary>
      <>
        <style>{`
          .scrollbar-hide::-webkit-scrollbar { display: none; }
          .scrollbar-hide { -ms-overflow-style: none; scrollbar-width: none; }
          .font-serif { font-family: 'Playfair Display', 'Georgia', serif; }
          @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
          .animate-fadeIn { animation: fadeIn 0.6s ease-out forwards; }
          @keyframes slideUp { from { opacity: 0; transform: translateY(30px); } to { opacity: 1; transform: translateY(0); } }
          .animate-slideUp { animation: slideUp 0.6s ease-out forwards; }

          /* Blob Animation */
          @keyframes blob {
            0% { transform: translate(0px, 0px) scale(1); }
            33% { transform: translate(30px, -50px) scale(1.1); }
            66% { transform: translate(-20px, 20px) scale(0.9); }
            100% { transform: translate(0px, 0px) scale(1); }
          }
          .animate-blob { animation: blob 7s infinite; }
          .animation-delay-2000 { animation-delay: 2s; }
          .animation-delay-4000 { animation-delay: 4s; }
        `}</style>

        {view === 'home' && <HomeScreen />}
        {view === 'loading' && <LoadingScreen />}
        {view === 'results' && <ResultsView />}
      </>
    </ErrorBoundary>
  );
}